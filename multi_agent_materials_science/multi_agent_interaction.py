from openai import OpenAI
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Composition
import os, json


RESEARCHER_BATCH_SYSTEM = """You are a battery materials researcher.
You:
- Propose promising cathode compositions for Li-ion batteries
- Target >20% higher capacity than LiCoO2 (~150–170 mAh/g)
- Care about safety and stability for wearable displays
Always respond ONLY with a JSON array of objects of the form:
[
  {"formula": "...", "reasoning": "..."},
  ...
]
Each "formula" must be a single Li–TM–O composition (e.g. "LiNi0.8Co0.1Mn0.1O2").
Return between 3 and 10 candidates as requested by the user."""


PROGRAMMER_SYSTEM = """You are a scientific programmer.
You receive:
- A target material formula
- A small dict of properties fetched from the Materials Project
You must:
- Explain what these properties mean for battery performance
- Comment on stability (energy_above_hull), band gap, density, etc.
- Say whether this looks like a promising cathode vs LiCoO2.
Be concise and quantitative where possible."""

from dataclasses import dataclass
from typing import List, Optional

BASELINE_LCO_CAPACITY_MAH_G = 165.0  

@dataclass
class CandidateMetrics:
    formula: str
    reasoning: str
    mp_props: Optional[dict]
    capacity_mAh_g: Optional[float]
    specific_energy_Wh_kg: Optional[float]
    volumetric_energy_Wh_L: Optional[float]
    validation_flags: dict
    anomalies: List[str]
    confidence: float


class BatteryMultiAgent:
    def __init__(self, openai_client: OpenAI, mp_api_key: str):
        self.client = openai_client
        self.mpr = MPRester(mp_api_key)


    def call_researcher(self, user_prompt: str) -> dict:
        resp = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": RESEARCHER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
            raise

    def call_programmer(self, formula: str, props: dict) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": PROGRAMMER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Formula: {formula}\n"
                        f"Properties: {json.dumps(props, indent=2)}"
                    ),
                },
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    def query_material_from_mp(self, formula: str, max_results: int = 5) -> dict | None:
        comp = Composition(formula)
        elements = sorted({el.symbol for el in comp.elements})

        chemsys = "-".join(elements)
        results = self.mpr.summary.search(chemsys=chemsys)

        if not results:
            return None

        best = min(results, key=lambda r: r.get("energy_above_hull", 1e9))

        return {
            "material_id": best.get("material_id"),
            "formula_pretty": best.get("formula_pretty"),
            "energy_above_hull_eV_per_atom": best.get("energy_above_hull"),
            "is_stable_flag": best.get("is_stable"),
            "band_gap_eV": best.get("band_gap"),
            "density_g_per_cm3": best.get("density"),
            "volume_A3": best.get("volume"),
            "nsites": best.get("nsites"),
            "possible_species": best.get("possible_species"),
            "chemsys": chemsys,
        }

    def one_iteration(self, baseline_note: str) -> None:
        researcher_out = self.call_researcher(
            "Our goal: cathode for wearable displays, >20% capacity vs LiCoO2.\n"
            f"Context: {baseline_note}\n"
            "Propose ONE promising Li–TM–O cathode composition."
        )
        formula = researcher_out["formula"]
        reasoning = researcher_out["reasoning"]

        print(f"\n🔬 Researcher proposed: {formula}")
        print(f"Reasoning: {reasoning}\n")

        props = self.query_material_from_mp(formula)
        if props is None:
            print("❌ No Materials Project entries found for this element set.")
            return

        print("📡 MP properties (compressed):")
        for k, v in props.items():
            print(f"  - {k}: {v}")
        print()

        programmer_comment = self.call_programmer(formula, props)
        print("💻 Programmer analysis:")
        print(programmer_comment)


    def call_researcher_batch(self, user_prompt: str, n_candidates: int = 5) -> List[dict]:
        """Ask the researcher agent for a batch of candidate formulas."""
        resp = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": RESEARCHER_BATCH_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal: cathode for wearable displays, >20% capacity vs LiCoO2.\n"
                        f"Context: {user_prompt}\n"
                        f"Propose {n_candidates} distinct Li–TM–O cathode compositions."
                    ),
                },
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
            else:
                raise

        if not isinstance(data, list):
            raise ValueError("Batch researcher output is not a list")

        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "formula" in item and "reasoning" in item:
                cleaned.append(item)
        return cleaned


    def compute_theoretical_capacity(self, formula):
        """
        Very simple approximation: assume 1 e- per Li.
        C_th ≈ 26.8 * (n_e / M) [mAh/g]
        where n_e = number of Li atoms per formula unit,
        M = molar mass [g/mol].
        """
        try:
            comp = Composition(formula)
            n_li = comp.get_el_amt_dict().get("Li", 0.0)
            if n_li <= 0:
                return None
            molar_mass = comp.weight  
            n_e = n_li  
            capacity = 26.801388888888887 * (n_e / molar_mass)  
            return capacity
        except Exception:
            return None

    def derive_energy_metrics(
        self,
        capacity_mAh_g: Optional[float],
        mp_props: Optional[dict],
        assumed_voltage_V: float = 3.7,
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute specific and volumetric energy using simple assumptions."""
        if capacity_mAh_g is None:
            return None, None

        specific_energy_Wh_kg = assumed_voltage_V * capacity_mAh_g / 1000.0  

        density = None
        if mp_props is not None:
            density = mp_props.get("density_g_per_cm3", None)

        volumetric_energy_Wh_L = None
        if density is not None:
            volumetric_energy_Wh_L = specific_energy_Wh_kg * density

        return specific_energy_Wh_kg, volumetric_energy_Wh_L

    def validate_and_score(
        self,
        formula: str,
        capacity_mAh_g: Optional[float],
        mp_props: Optional[dict],
        baseline_capacity_mAh_g: float = BASELINE_LCO_CAPACITY_MAH_G,
    ) -> tuple[dict, List[str], float]:
        """
        Multi-tier validation + very simple confidence model.
        Returns: (validation_flags, anomalies, confidence)
        """
        flags = {
            "has_mp_data": mp_props is not None,
            "stable_ehull": None,
            "bandgap_ok": None,
            "capacity_gain_ok": None,
        }
        anomalies: List[str] = []

        if mp_props is None:

            return flags, ["no_mp_data"], 0.2

        ehull = mp_props.get("energy_above_hull_eV_per_atom", None)
        bandgap = mp_props.get("band_gap_eV", None)

        if ehull is not None:
            flags["stable_ehull"] = ehull < 0.03
            if ehull > 0.05:
                anomalies.append("high_energy_above_hull")
        else:
            anomalies.append("missing_ehull")

        if bandgap is not None:
            flags["bandgap_ok"] = 0.0 <= bandgap <= 2.0
            if bandgap == 0.0:
                anomalies.append("metallic_bandgap_zero")
            if bandgap > 3.0:
                anomalies.append("insulating_high_bandgap")
        else:
            anomalies.append("missing_bandgap")

        if capacity_mAh_g is not None:
            flags["capacity_gain_ok"] = capacity_mAh_g >= 1.2 * baseline_capacity_mAh_g
        else:
            anomalies.append("missing_capacity_estimate")

        confidence = 0.3  
        if mp_props is not None:
            confidence += 0.2
        if flags["stable_ehull"] is True:
            confidence += 0.2
        if flags["bandgap_ok"] is True:
            confidence += 0.1
        if flags["capacity_gain_ok"] is True:
            confidence += 0.1
        if "high_energy_above_hull" in anomalies:
            confidence -= 0.2
        if "no_mp_data" in anomalies:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))
        return flags, anomalies, confidence


    def screen_candidates(
        self,
        baseline_note: str,
        n_candidates: int = 5,
    ) -> List[CandidateMetrics]:
        """
        Full multi-candidate high-throughput screening:
        - Researcher proposes N candidates
        - MP queried for each
        - Capacities + energy metrics derived
        - Multi-tier validation + confidence computed
        - Ranked report printed
        """
        batch = self.call_researcher_batch(
            baseline_note,
            n_candidates=n_candidates,
        )

        print(f"\n🔬 Researcher proposed {len(batch)} candidates:\n")
        for i, c in enumerate(batch, start=1):
            print(f"  [{i}] {c['formula']}: {c['reasoning']}")
        print()

        results: List[CandidateMetrics] = []

        for item in batch:
            formula = item["formula"]
            reasoning = item["reasoning"]

            print(f"\n🔎 Screening candidate: {formula}")
            mp_props = self.query_material_from_mp(formula)

            if mp_props is None:
                print("  ❌ No MP entries found.")
            else:
                print("  📡 MP summary:")
                for k, v in mp_props.items():
                    print(f"    - {k}: {v}")

            capacity = self.compute_theoretical_capacity(formula)
            spec_E, vol_E = self.derive_energy_metrics(capacity, mp_props)
            flags, anomalies, conf = self.validate_and_score(formula, capacity, mp_props)

            print("  📊 Derived metrics:")
            print(f"    - capacity_mAh_g: {capacity}")
            print(f"    - specific_energy_Wh_kg: {spec_E}")
            print(f"    - volumetric_energy_Wh_L: {vol_E}")
            print(f"    - validation_flags: {flags}")
            print(f"    - anomalies: {anomalies}")
            print(f"    - confidence: {conf:.2f}")

            results.append(
                CandidateMetrics(
                    formula=formula,
                    reasoning=reasoning,
                    mp_props=mp_props,
                    capacity_mAh_g=capacity,
                    specific_energy_Wh_kg=spec_E,
                    volumetric_energy_Wh_L=vol_E,
                    validation_flags=flags,
                    anomalies=anomalies,
                    confidence=conf,
                )
            )

        def score_for_sort(c):
            cap = c.capacity_mAh_g or 0.0
            stable_bonus = 20.0 if c.validation_flags.get("stable_ehull") else 0.0
            conf_bonus = 50.0 * c.confidence
            return cap + stable_bonus + conf_bonus

        results_sorted = sorted(results, key=score_for_sort, reverse=True)

        print("\n🏁 Final ranked candidates (best to worst):\n")
        print(
            f"{'Rank':<4} {'Formula':<20} {'Cap(mAh/g)':>12} "
            f"{'E_hull(eV)':>12} {'BandGap(eV)':>12} {'Conf':>6} {'Flags':>10}"
        )
        for idx, c in enumerate(results_sorted, start=1):
            ehull = None
            bandgap = None
            if c.mp_props:
                ehull = c.mp_props.get("energy_above_hull_eV_per_atom", None)
                bandgap = c.mp_props.get("band_gap_eV", None)

            print(
                f"{idx:<4} {c.formula:<20} "
                f"{(c.capacity_mAh_g or float('nan')):>12.1f} "
                f"{(ehull if ehull is not None else float('nan')):>12.3f} "
                f"{(bandgap if bandgap is not None else float('nan')):>12.3f} "
                f"{c.confidence:>6.2f} "
                f"{str(c.validation_flags):>10}"
            )

        print("\n🚨 Human-review flags (if any):")
        for c in results_sorted:
            if ("no_mp_data" in c.anomalies) or ("high_energy_above_hull" in c.anomalies):
                print(f"  - {c.formula}: anomalies={c.anomalies}, conf={c.confidence:.2f}")

        return results_sorted


openai_client = OpenAI(api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

mp_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  
if not mp_key:
    raise RuntimeError("Set your Materials Project API key in MP_API_KEY or MAPI_KEY")

agent = BatteryMultiAgent(openai_client, mp_key)

baseline = "LiCoO2 practical capacity ≈ 150–170 mAh/g, ~3.9 V vs Li/Li+, good but Co is costly/toxic."


agent.screen_candidates(baseline_note=baseline, n_candidates=5)
