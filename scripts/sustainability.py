"""Per-GPU energy, carbon and water, computed from the spec sheet.

`_meta` already described this section as "per-GPU figures computed from TDP".
It was not: gpu_carbon was a hand-written table, and it had drifted from specs
in the way hand-written tables do.

GB200 was the visible failure. Its row carried 2700W -- the TDP of the whole
Grace Blackwell superchip, two Blackwell GPUs plus a Grace CPU -- while every
other GB200 figure on the dashboard is per GPU: the price is $/GPU-hour, VRAM
is 186GB per GPU, and specs.tdp_watts says 1200. So GB200's power draw, its
annual carbon, its water use and the self-hosted power line in its TCO were
all 2.25x too high.

The table also only covered 9 of 17 tracked GPUs, so RTX-5090, L40S, GH200,
L4, T4, H100-PCIe, A100-40GB and MI250X were simply absent from the
sustainability tab.

Computing it from specs each run fixes both, and means it cannot drift again.
The constants below are the only inputs that are not arithmetic; each is
stated with its source and its uncertainty.
"""

# Sustained draw as a share of nameplate TDP. A GPU under continuous training
# or inference does not sit at its power limit -- measured sustained draw runs
# around 80-85% of TDP. The previous table used 0.82-0.83 for every part except
# B300, which sat at 0.70 for no stated reason.
SUSTAINED_LOAD_FACTOR = 0.82

# US average grid carbon intensity, EPA eGRID (~0.40 kg CO2e/kWh). Matches the
# 0.400 implied by every row of the previous table.
US_GRID_KG_CO2_PER_KWH = 0.40

# Nordic grid (hydro/wind dominated), ~45 g CO2e/kWh. The previous table
# implied 0.112 x US, i.e. 0.045 -- kept.
NORDIC_GRID_KG_CO2_PER_KWH = 0.045

# Datacenter water use effectiveness, litres per kWh delivered. US industry
# average is ~1.8 L/kWh; the previous table implied exactly this.
WATER_L_PER_KWH = 1.8

# Embodied (manufacturing) carbon per kW of nameplate TDP. The old per-GPU
# figures implied 130-250 kg/kW with no method behind the spread, so this uses
# the midpoint as a single documented rate. It is the roughest number here and
# is labelled as an order-of-magnitude estimate, not a measurement.
EMBODIED_KG_CO2_PER_KW = 200


def build_gpu_carbon(data, log_ok=lambda a, b="": None):
    """Recompute sustainability.gpu_carbon for every tracked GPU from specs."""
    specs = data.get("specs") or {}
    if not specs:
        return data

    out = {}
    for gpu_id, spec in specs.items():
        tdp = spec.get("tdp_watts")
        if not tdp:
            continue
        typical = round(tdp * SUSTAINED_LOAD_FACTOR)
        kwh_hr = round(typical / 1000, 3)
        annual_kwh = round(kwh_hr * 8760)
        out[gpu_id] = {
            "tdp_watts": tdp,
            "typical_watts": typical,
            "kwh_per_hour": kwh_hr,
            "annual_kwh_full_util": annual_kwh,
            "carbon_kg_per_year_us_avg": round(annual_kwh * US_GRID_KG_CO2_PER_KWH),
            "carbon_kg_per_year_eu_nordic": round(annual_kwh * NORDIC_GRID_KG_CO2_PER_KWH),
            "water_liters_per_year_us_avg": round(annual_kwh * WATER_L_PER_KWH),
            "embodied_carbon_kg": round(tdp / 1000 * EMBODIED_KG_CO2_PER_KW),
        }

    sustainability = data.setdefault("sustainability", {})
    sustainability["gpu_carbon"] = out
    sustainability["assumptions"] = {
        "sustained_load_factor": SUSTAINED_LOAD_FACTOR,
        "us_grid_kg_co2_per_kwh": US_GRID_KG_CO2_PER_KWH,
        "nordic_grid_kg_co2_per_kwh": NORDIC_GRID_KG_CO2_PER_KWH,
        "water_l_per_kwh": WATER_L_PER_KWH,
        "embodied_kg_co2_per_kw": EMBODIED_KG_CO2_PER_KW,
        "note": (
            "Everything in gpu_carbon is these five constants applied to "
            "specs.tdp_watts. Figures are per GPU, matching how price and VRAM "
            "are stated everywhere else on this dashboard -- a multi-GPU module "
            "like GB200 is charged for its own share, not the whole superchip."
        ),
    }

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["sustainability"] = {
        "basis": "derived",
        "detail": (
            "Per-GPU energy, carbon and water computed from specs.tdp_watts "
            "each run, using the five constants in sustainability.assumptions "
            "(sustained load 82% of TDP, US grid 0.40 kg CO2e/kWh, Nordic "
            "0.045, water 1.8 L/kWh, embodied 200 kg/kW). Previously a "
            "hand-written table that covered 9 of 17 GPUs and had drifted from "
            "specs -- GB200 carried the 2700W superchip TDP against a per-GPU "
            "price, overstating its power and carbon 2.25x. Embodied carbon is "
            "an order-of-magnitude estimate; the rest is arithmetic. Provider "
            "PUE and grid figures alongside this are from published "
            "sustainability reports."
        ),
    }
    log_ok("Sustainability", f"{len(out)} GPUs computed from specs")
    return data
