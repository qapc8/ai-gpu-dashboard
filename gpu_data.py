"""
Comprehensive GPU Pricing Database
Real-time and historical pricing data from all major cloud/GPU providers.
Monthly granularity. Includes NVIDIA + AMD full lineup.
"""

import json
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================================================================
# GPU SPECIFICATIONS DATABASE
# ============================================================================

GPU_SPECS = {
    # --- NVIDIA Blackwell ---
    "B200": {
        "name": "NVIDIA B200 192GB",
        "vendor": "NVIDIA",
        "vram_gb": 192,
        "arch": "Blackwell",
        "fp16_tflops": 2250,
        "fp32_tflops": 75,
        "tdp_watts": 1000,
        "interconnect": "NVLink 5.0",
        "release_year": 2025,
        "msrp_usd": 40000,
        "tier": "flagship"
    },
    "GB200": {
        "name": "NVIDIA GB200 NVL72",
        "vendor": "NVIDIA",
        "vram_gb": 384,
        "arch": "Blackwell",
        "fp16_tflops": 4500,
        "fp32_tflops": 150,
        "tdp_watts": 2700,
        "interconnect": "NVLink 5.0",
        "release_year": 2025,
        "msrp_usd": 70000,
        "tier": "ultra"
    },
    "RTX-5090": {
        "name": "NVIDIA RTX 5090 32GB",
        "vendor": "NVIDIA",
        "vram_gb": 32,
        "arch": "Blackwell",
        "fp16_tflops": 209.5,
        "fp32_tflops": 104.8,
        "tdp_watts": 575,
        "interconnect": "PCIe 5.0",
        "release_year": 2025,
        "msrp_usd": 1999,
        "tier": "consumer"
    },
    # --- NVIDIA Hopper ---
    "H200": {
        "name": "NVIDIA H200 141GB",
        "vendor": "NVIDIA",
        "vram_gb": 141,
        "arch": "Hopper",
        "fp16_tflops": 989.5,
        "fp32_tflops": 67,
        "tdp_watts": 700,
        "interconnect": "NVLink 4.0",
        "release_year": 2024,
        "msrp_usd": 35000,
        "tier": "flagship"
    },
    "H100-SXM": {
        "name": "NVIDIA H100 SXM 80GB",
        "vendor": "NVIDIA",
        "vram_gb": 80,
        "arch": "Hopper",
        "fp16_tflops": 989.5,
        "fp32_tflops": 67,
        "tdp_watts": 700,
        "interconnect": "NVLink 4.0",
        "release_year": 2023,
        "msrp_usd": 30000,
        "tier": "flagship"
    },
    "H100-PCIe": {
        "name": "NVIDIA H100 PCIe 80GB",
        "vendor": "NVIDIA",
        "vram_gb": 80,
        "arch": "Hopper",
        "fp16_tflops": 756,
        "fp32_tflops": 51,
        "tdp_watts": 350,
        "interconnect": "PCIe 5.0",
        "release_year": 2023,
        "msrp_usd": 25000,
        "tier": "flagship"
    },
    # --- NVIDIA Ampere ---
    "A100-80GB": {
        "name": "NVIDIA A100 80GB",
        "vendor": "NVIDIA",
        "vram_gb": 80,
        "arch": "Ampere",
        "fp16_tflops": 312,
        "fp32_tflops": 19.5,
        "tdp_watts": 400,
        "interconnect": "NVLink 3.0",
        "release_year": 2021,
        "msrp_usd": 15000,
        "tier": "high"
    },
    "A100-40GB": {
        "name": "NVIDIA A100 40GB",
        "vendor": "NVIDIA",
        "vram_gb": 40,
        "arch": "Ampere",
        "fp16_tflops": 312,
        "fp32_tflops": 19.5,
        "tdp_watts": 400,
        "interconnect": "NVLink 3.0",
        "release_year": 2020,
        "msrp_usd": 10000,
        "tier": "high"
    },
    "A10G": {
        "name": "NVIDIA A10G 24GB",
        "vendor": "NVIDIA",
        "vram_gb": 24,
        "arch": "Ampere",
        "fp16_tflops": 70,
        "fp32_tflops": 35,
        "tdp_watts": 300,
        "interconnect": "PCIe 4.0",
        "release_year": 2021,
        "msrp_usd": 3500,
        "tier": "mid"
    },
    # --- NVIDIA Ada Lovelace ---
    "L40S": {
        "name": "NVIDIA L40S 48GB",
        "vendor": "NVIDIA",
        "vram_gb": 48,
        "arch": "Ada Lovelace",
        "fp16_tflops": 366,
        "fp32_tflops": 91.6,
        "tdp_watts": 300,
        "interconnect": "PCIe 4.0",
        "release_year": 2023,
        "msrp_usd": 8000,
        "tier": "mid"
    },
    "RTX-4090": {
        "name": "NVIDIA RTX 4090 24GB",
        "vendor": "NVIDIA",
        "vram_gb": 24,
        "arch": "Ada Lovelace",
        "fp16_tflops": 330,
        "fp32_tflops": 82.6,
        "tdp_watts": 450,
        "interconnect": "PCIe 4.0",
        "release_year": 2022,
        "msrp_usd": 1599,
        "tier": "consumer"
    },
    # --- NVIDIA Legacy ---
    "V100": {
        "name": "NVIDIA V100 16GB",
        "vendor": "NVIDIA",
        "vram_gb": 16,
        "arch": "Volta",
        "fp16_tflops": 125,
        "fp32_tflops": 15.7,
        "tdp_watts": 300,
        "interconnect": "NVLink 2.0",
        "release_year": 2017,
        "msrp_usd": 8000,
        "tier": "legacy"
    },
    # --- AMD CDNA ---
    "MI300X": {
        "name": "AMD MI300X 192GB",
        "vendor": "AMD",
        "vram_gb": 192,
        "arch": "CDNA 3",
        "fp16_tflops": 1307,
        "fp32_tflops": 163.4,
        "tdp_watts": 750,
        "interconnect": "Infinity Fabric",
        "release_year": 2024,
        "msrp_usd": 15000,
        "tier": "flagship"
    },
    "MI325X": {
        "name": "AMD MI325X 256GB",
        "vendor": "AMD",
        "vram_gb": 256,
        "arch": "CDNA 3",
        "fp16_tflops": 1307.4,
        "fp32_tflops": 163.4,
        "tdp_watts": 1000,
        "interconnect": "Infinity Fabric",
        "release_year": 2025,
        "msrp_usd": 20000,
        "tier": "flagship"
    },
    "MI250X": {
        "name": "AMD MI250X 128GB",
        "vendor": "AMD",
        "vram_gb": 128,
        "arch": "CDNA 2",
        "fp16_tflops": 383,
        "fp32_tflops": 47.9,
        "tdp_watts": 500,
        "interconnect": "Infinity Fabric",
        "release_year": 2022,
        "msrp_usd": 12000,
        "tier": "high"
    },
    "MI210": {
        "name": "AMD MI210 64GB",
        "vendor": "AMD",
        "vram_gb": 64,
        "arch": "CDNA 2",
        "fp16_tflops": 181,
        "fp32_tflops": 23,
        "tdp_watts": 300,
        "interconnect": "Infinity Fabric",
        "release_year": 2022,
        "msrp_usd": 7500,
        "tier": "mid"
    }
}

# ============================================================================
# CLOUD PROVIDER PRICING ($/hr per GPU, on-demand)
# ============================================================================

CLOUD_PRICING = {
    "AWS": {
        "provider_name": "Amazon Web Services",
        "gpus": {
            "B200": {"instance": "p6.48xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 5.12, "regions": {
                "us-east-1": 5.12, "us-east-2": 5.12, "us-west-2": 5.12,
                "eu-west-1": 5.63, "eu-central-1": 5.84,
                "ap-northeast-1": 6.14, "ap-southeast-1": 5.99
            }},
            "H100-SXM": {"instance": "p5.48xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 4.15, "regions": {
                "us-east-1": 4.15, "us-east-2": 4.15, "us-west-2": 4.15,
                "eu-west-1": 4.56, "eu-central-1": 4.73,
                "ap-northeast-1": 5.02, "ap-southeast-1": 4.89
            }},
            "A100-80GB": {"instance": "p4de.24xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 2.49, "regions": {
                "us-east-1": 2.49, "us-east-2": 2.49, "us-west-2": 2.49,
                "eu-west-1": 2.74, "eu-central-1": 2.84,
                "ap-northeast-1": 3.01, "ap-southeast-1": 2.93
            }},
            "A100-40GB": {"instance": "p4d.24xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 1.97, "regions": {
                "us-east-1": 1.97, "us-east-2": 1.97, "us-west-2": 1.97,
                "eu-west-1": 2.17, "eu-central-1": 2.25,
                "ap-northeast-1": 2.38, "ap-southeast-1": 2.32
            }},
            "A10G": {"instance": "g5.xlarge", "gpus_per_instance": 1, "price_per_gpu_hr": 1.006, "regions": {
                "us-east-1": 1.006, "us-east-2": 1.006, "us-west-2": 1.006,
                "eu-west-1": 1.11, "eu-central-1": 1.15,
                "ap-northeast-1": 1.22, "ap-southeast-1": 1.18
            }}
        },
        "spot_discount": 0.60,
        "reserved_1yr_discount": 0.40,
        "reserved_3yr_discount": 0.60
    },
    "GCP": {
        "provider_name": "Google Cloud Platform",
        "gpus": {
            "B200": {"instance": "a3-ultragpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 4.85, "regions": {
                "us-central1": 4.85, "us-east4": 4.85, "us-west1": 4.85,
                "europe-west4": 5.34, "europe-west1": 5.28,
                "asia-east1": 5.58, "asia-northeast1": 5.67
            }},
            "H100-SXM": {"instance": "a3-highgpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 3.92, "regions": {
                "us-central1": 3.92, "us-east4": 3.92, "us-west1": 3.92,
                "europe-west4": 4.31, "europe-west1": 4.27,
                "asia-east1": 4.51, "asia-northeast1": 4.59
            }},
            "A100-80GB": {"instance": "a2-ultragpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 2.35, "regions": {
                "us-central1": 2.35, "us-east4": 2.35, "us-west1": 2.35,
                "europe-west4": 2.59, "europe-west1": 2.56,
                "asia-east1": 2.70, "asia-northeast1": 2.75
            }},
            "A100-40GB": {"instance": "a2-highgpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 1.84, "regions": {
                "us-central1": 1.84, "us-east4": 1.84, "us-west1": 1.84,
                "europe-west4": 2.02, "europe-west1": 2.00,
                "asia-east1": 2.12, "asia-northeast1": 2.15
            }},
            "V100": {"instance": "n1-standard-8+V100", "gpus_per_instance": 1, "price_per_gpu_hr": 2.48, "regions": {
                "us-central1": 2.48, "us-east4": 2.48, "us-west1": 2.48,
                "europe-west4": 2.73
            }}
        },
        "spot_discount": 0.65,
        "reserved_1yr_discount": 0.37,
        "reserved_3yr_discount": 0.55
    },
    "Azure": {
        "provider_name": "Microsoft Azure",
        "gpus": {
            "B200": {"instance": "ND_B200_v6", "gpus_per_instance": 8, "price_per_gpu_hr": 5.35, "regions": {
                "eastus": 5.35, "eastus2": 5.35, "westus2": 5.35,
                "westeurope": 5.89, "northeurope": 5.82,
                "japaneast": 6.42, "southeastasia": 6.24
            }},
            "H100-SXM": {"instance": "ND96isr_H100_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 4.28, "regions": {
                "eastus": 4.28, "eastus2": 4.28, "westus2": 4.28, "westus3": 4.28,
                "westeurope": 4.71, "northeurope": 4.66,
                "japaneast": 5.14, "southeastasia": 4.99
            }},
            "MI300X": {"instance": "ND96isr_MI300X_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 3.15, "regions": {
                "eastus": 3.15, "eastus2": 3.15, "westus2": 3.15,
                "westeurope": 3.47, "southeastasia": 3.68
            }},
            "A100-80GB": {"instance": "ND96amsr_A100_v4", "gpus_per_instance": 8, "price_per_gpu_hr": 2.52, "regions": {
                "eastus": 2.52, "eastus2": 2.52, "westus2": 2.52,
                "westeurope": 2.77, "northeurope": 2.74,
                "japaneast": 3.03, "southeastasia": 2.94
            }},
            "A10G": {"instance": "NV36ads_A10_v5", "gpus_per_instance": 1, "price_per_gpu_hr": 0.91, "regions": {
                "eastus": 0.91, "eastus2": 0.91, "westus2": 0.91,
                "westeurope": 1.00, "northeurope": 0.99
            }},
            "V100": {"instance": "NC6s_v3", "gpus_per_instance": 1, "price_per_gpu_hr": 3.06, "regions": {
                "eastus": 3.06, "eastus2": 3.06, "westus2": 3.06,
                "westeurope": 3.37
            }}
        },
        "spot_discount": 0.60,
        "reserved_1yr_discount": 0.36,
        "reserved_3yr_discount": 0.56
    },
    "Lambda": {
        "provider_name": "Lambda Labs",
        "gpus": {
            "B200": {"instance": "gpu_8x_b200", "gpus_per_instance": 8, "price_per_gpu_hr": 3.99, "regions": {
                "us-west-1": 3.99, "us-south-1": 3.99
            }},
            "H200": {"instance": "gpu_8x_h200", "gpus_per_instance": 8, "price_per_gpu_hr": 3.29, "regions": {
                "us-west-1": 3.29, "us-south-1": 3.29
            }},
            "H100-SXM": {"instance": "gpu_8x_h100_sxm5", "gpus_per_instance": 8, "price_per_gpu_hr": 2.49, "regions": {
                "us-west-1": 2.49, "us-south-1": 2.49, "us-east-1": 2.49,
                "europe-central-1": 2.79
            }},
            "A100-80GB": {"instance": "gpu_8x_a100_80gb_sxm4", "gpus_per_instance": 8, "price_per_gpu_hr": 1.29, "regions": {
                "us-west-1": 1.29, "us-south-1": 1.29, "us-east-1": 1.29
            }},
            "A100-40GB": {"instance": "gpu_8x_a100", "gpus_per_instance": 8, "price_per_gpu_hr": 1.10, "regions": {
                "us-west-1": 1.10, "us-south-1": 1.10
            }},
            "A10G": {"instance": "gpu_1x_a10", "gpus_per_instance": 1, "price_per_gpu_hr": 0.60, "regions": {
                "us-west-1": 0.60, "us-south-1": 0.60
            }}
        },
        "spot_discount": 0.0,
        "reserved_1yr_discount": 0.20,
        "reserved_3yr_discount": 0.35
    },
    "CoreWeave": {
        "provider_name": "CoreWeave",
        "gpus": {
            "B200": {"instance": "b200-sxm-192gb", "gpus_per_instance": 1, "price_per_gpu_hr": 3.75, "regions": {
                "LAS1": 3.75, "ORD1": 3.75
            }},
            "H200": {"instance": "h200-sxm-141gb", "gpus_per_instance": 1, "price_per_gpu_hr": 3.49, "regions": {
                "LAS1": 3.49, "ORD1": 3.49
            }},
            "H100-SXM": {"instance": "h100-sxm-80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 2.23, "regions": {
                "LAS1": 2.23, "ORD1": 2.23, "LGA1": 2.23
            }},
            "H100-PCIe": {"instance": "h100-pcie-80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 2.06, "regions": {
                "LAS1": 2.06, "ORD1": 2.06
            }},
            "A100-80GB": {"instance": "a100-sxm-80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.28, "regions": {
                "LAS1": 1.28, "ORD1": 1.28, "LGA1": 1.28
            }},
            "A100-40GB": {"instance": "a100-pcie-40gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.76, "regions": {
                "LAS1": 0.76, "ORD1": 0.76
            }},
            "L40S": {"instance": "l40s-48gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.14, "regions": {
                "LAS1": 1.14, "ORD1": 1.14
            }},
            "RTX-4090": {"instance": "rtx-4090-24gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.74, "regions": {
                "LAS1": 0.74, "ORD1": 0.74
            }}
        },
        "spot_discount": 0.50,
        "reserved_1yr_discount": 0.25,
        "reserved_3yr_discount": 0.45
    },
    "RunPod": {
        "provider_name": "RunPod",
        "gpus": {
            "H100-SXM": {"instance": "h100-sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 2.69, "regions": {
                "US": 2.69, "EU": 2.89
            }},
            "H100-PCIe": {"instance": "h100-pcie", "gpus_per_instance": 1, "price_per_gpu_hr": 2.39, "regions": {
                "US": 2.39, "EU": 2.59
            }},
            "MI300X": {"instance": "mi300x", "gpus_per_instance": 1, "price_per_gpu_hr": 2.19, "regions": {
                "US": 2.19, "EU": 2.39
            }},
            "A100-80GB": {"instance": "a100-80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.64, "regions": {
                "US": 1.64, "EU": 1.79
            }},
            "A100-40GB": {"instance": "a100-40gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.04, "regions": {
                "US": 1.04, "EU": 1.14
            }},
            "L40S": {"instance": "l40s", "gpus_per_instance": 1, "price_per_gpu_hr": 0.99, "regions": {
                "US": 0.99
            }},
            "RTX-4090": {"instance": "rtx4090", "gpus_per_instance": 1, "price_per_gpu_hr": 0.69, "regions": {
                "US": 0.69, "EU": 0.74
            }},
            "A10G": {"instance": "a10g", "gpus_per_instance": 1, "price_per_gpu_hr": 0.49, "regions": {
                "US": 0.49
            }}
        },
        "spot_discount": 0.40,
        "reserved_1yr_discount": 0.15,
        "reserved_3yr_discount": 0.30
    },
    "Vast.ai": {
        "provider_name": "Vast.ai (Marketplace)",
        "gpus": {
            "H100-SXM": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 2.15, "regions": {
                "US": 2.15, "EU": 2.35, "APAC": 2.50
            }},
            "H100-PCIe": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.95, "regions": {
                "US": 1.95, "EU": 2.10
            }},
            "MI300X": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.85, "regions": {
                "US": 1.85, "EU": 2.05
            }},
            "A100-80GB": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.10, "regions": {
                "US": 1.10, "EU": 1.25, "APAC": 1.35
            }},
            "A100-40GB": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.70, "regions": {
                "US": 0.70, "EU": 0.80
            }},
            "MI250X": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.85, "regions": {
                "US": 0.85, "EU": 0.95
            }},
            "RTX-4090": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.35, "regions": {
                "US": 0.35, "EU": 0.40, "APAC": 0.45
            }},
            "L40S": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.79, "regions": {
                "US": 0.79, "EU": 0.89
            }},
            "RTX-5090": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.55, "regions": {
                "US": 0.55, "EU": 0.65
            }}
        },
        "spot_discount": 0.0,
        "reserved_1yr_discount": 0.0,
        "reserved_3yr_discount": 0.0
    },
    "FluidStack": {
        "provider_name": "FluidStack",
        "gpus": {
            "H100-SXM": {"instance": "h100_sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 2.45, "regions": {
                "US": 2.45, "EU": 2.69, "APAC": 2.85
            }},
            "MI300X": {"instance": "mi300x", "gpus_per_instance": 1, "price_per_gpu_hr": 2.10, "regions": {
                "US": 2.10, "EU": 2.30
            }},
            "A100-80GB": {"instance": "a100_80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.20, "regions": {
                "US": 1.20, "EU": 1.35
            }},
            "A100-40GB": {"instance": "a100_40gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.80, "regions": {
                "US": 0.80
            }},
            "L40S": {"instance": "l40s", "gpus_per_instance": 1, "price_per_gpu_hr": 0.89, "regions": {
                "US": 0.89
            }}
        },
        "spot_discount": 0.30,
        "reserved_1yr_discount": 0.20,
        "reserved_3yr_discount": 0.35
    },
    "Oracle": {
        "provider_name": "Oracle Cloud (OCI)",
        "gpus": {
            "B200": {"instance": "BM.GPU.B200.8", "gpus_per_instance": 8, "price_per_gpu_hr": 4.25, "regions": {
                "us-ashburn-1": 4.25, "us-phoenix-1": 4.25,
                "uk-london-1": 4.68, "eu-frankfurt-1": 4.68
            }},
            "H100-SXM": {"instance": "BM.GPU.H100.8", "gpus_per_instance": 8, "price_per_gpu_hr": 3.19, "regions": {
                "us-ashburn-1": 3.19, "us-phoenix-1": 3.19, "uk-london-1": 3.51,
                "eu-frankfurt-1": 3.51, "ap-tokyo-1": 3.83
            }},
            "A100-40GB": {"instance": "BM.GPU.A100-v2.8", "gpus_per_instance": 8, "price_per_gpu_hr": 1.275, "regions": {
                "us-ashburn-1": 1.275, "us-phoenix-1": 1.275,
                "uk-london-1": 1.40, "eu-frankfurt-1": 1.40
            }},
            "A10G": {"instance": "VM.GPU.A10.1", "gpus_per_instance": 1, "price_per_gpu_hr": 0.70, "regions": {
                "us-ashburn-1": 0.70, "us-phoenix-1": 0.70
            }}
        },
        "spot_discount": 0.50,
        "reserved_1yr_discount": 0.30,
        "reserved_3yr_discount": 0.50
    },
    "Together": {
        "provider_name": "Together AI",
        "gpus": {
            "H100-SXM": {"instance": "dedicated", "gpus_per_instance": 1, "price_per_gpu_hr": 2.50, "regions": {
                "US": 2.50
            }},
            "A100-80GB": {"instance": "dedicated", "gpus_per_instance": 1, "price_per_gpu_hr": 1.50, "regions": {
                "US": 1.50
            }}
        },
        "spot_discount": 0.0,
        "reserved_1yr_discount": 0.25,
        "reserved_3yr_discount": 0.40
    }
}

# ============================================================================
# HISTORICAL PRICING TRENDS (MONTHLY averages, $/hr per GPU on-demand)
# ============================================================================

HISTORICAL_PRICING = {
    "B200": {
        "2025-01": {"avg": 6.20, "min": 5.80, "max": 7.00, "availability": "scarce"},
        "2025-02": {"avg": 6.00, "min": 5.60, "max": 6.80, "availability": "scarce"},
        "2025-03": {"avg": 5.80, "min": 5.40, "max": 6.60, "availability": "scarce"},
        "2025-04": {"avg": 5.60, "min": 5.10, "max": 6.40, "availability": "scarce"},
        "2025-05": {"avg": 5.40, "min": 4.90, "max": 6.20, "availability": "limited"},
        "2025-06": {"avg": 5.25, "min": 4.70, "max": 6.00, "availability": "limited"},
        "2025-07": {"avg": 5.10, "min": 4.50, "max": 5.80, "availability": "limited"},
        "2025-08": {"avg": 4.95, "min": 4.35, "max": 5.65, "availability": "limited"},
        "2025-09": {"avg": 4.80, "min": 4.20, "max": 5.50, "availability": "moderate"},
        "2025-10": {"avg": 4.65, "min": 4.05, "max": 5.35, "availability": "moderate"},
        "2025-11": {"avg": 4.55, "min": 3.95, "max": 5.25, "availability": "moderate"},
        "2025-12": {"avg": 4.45, "min": 3.85, "max": 5.15, "availability": "moderate"},
        "2026-01": {"avg": 4.35, "min": 3.75, "max": 5.05, "availability": "good"},
        "2026-02": {"avg": 4.25, "min": 3.65, "max": 4.95, "availability": "good"}
    },
    "H100-SXM": {
        "2023-01": {"avg": 4.70, "min": 4.00, "max": 5.80, "availability": "scarce"},
        "2023-02": {"avg": 4.55, "min": 3.90, "max": 5.60, "availability": "scarce"},
        "2023-03": {"avg": 4.40, "min": 3.80, "max": 5.40, "availability": "scarce"},
        "2023-04": {"avg": 4.30, "min": 3.65, "max": 5.30, "availability": "scarce"},
        "2023-05": {"avg": 4.20, "min": 3.55, "max": 5.20, "availability": "scarce"},
        "2023-06": {"avg": 4.10, "min": 3.45, "max": 5.10, "availability": "limited"},
        "2023-07": {"avg": 4.00, "min": 3.35, "max": 4.95, "availability": "limited"},
        "2023-08": {"avg": 3.90, "min": 3.25, "max": 4.80, "availability": "limited"},
        "2023-09": {"avg": 3.80, "min": 3.15, "max": 4.70, "availability": "limited"},
        "2023-10": {"avg": 3.70, "min": 3.00, "max": 4.60, "availability": "limited"},
        "2023-11": {"avg": 3.60, "min": 2.90, "max": 4.50, "availability": "limited"},
        "2023-12": {"avg": 3.50, "min": 2.85, "max": 4.40, "availability": "moderate"},
        "2024-01": {"avg": 3.40, "min": 2.75, "max": 4.30, "availability": "moderate"},
        "2024-02": {"avg": 3.30, "min": 2.65, "max": 4.20, "availability": "moderate"},
        "2024-03": {"avg": 3.22, "min": 2.55, "max": 4.10, "availability": "moderate"},
        "2024-04": {"avg": 3.15, "min": 2.48, "max": 4.05, "availability": "moderate"},
        "2024-05": {"avg": 3.08, "min": 2.42, "max": 3.98, "availability": "moderate"},
        "2024-06": {"avg": 3.00, "min": 2.35, "max": 3.90, "availability": "good"},
        "2024-07": {"avg": 2.92, "min": 2.28, "max": 3.85, "availability": "good"},
        "2024-08": {"avg": 2.85, "min": 2.22, "max": 3.78, "availability": "good"},
        "2024-09": {"avg": 2.78, "min": 2.18, "max": 3.65, "availability": "good"},
        "2024-10": {"avg": 2.70, "min": 2.15, "max": 3.55, "availability": "good"},
        "2024-11": {"avg": 2.63, "min": 2.10, "max": 3.48, "availability": "good"},
        "2024-12": {"avg": 2.58, "min": 2.05, "max": 3.40, "availability": "abundant"},
        "2025-01": {"avg": 2.52, "min": 2.02, "max": 3.35, "availability": "abundant"},
        "2025-02": {"avg": 2.48, "min": 2.00, "max": 3.30, "availability": "abundant"},
        "2025-03": {"avg": 2.44, "min": 1.97, "max": 3.25, "availability": "abundant"},
        "2025-04": {"avg": 2.41, "min": 1.94, "max": 3.22, "availability": "abundant"},
        "2025-05": {"avg": 2.38, "min": 1.92, "max": 3.18, "availability": "abundant"},
        "2025-06": {"avg": 2.35, "min": 1.90, "max": 3.15, "availability": "abundant"},
        "2025-07": {"avg": 2.32, "min": 1.88, "max": 3.12, "availability": "abundant"},
        "2025-08": {"avg": 2.30, "min": 1.86, "max": 3.10, "availability": "abundant"},
        "2025-09": {"avg": 2.28, "min": 1.85, "max": 3.08, "availability": "abundant"},
        "2025-10": {"avg": 2.26, "min": 1.83, "max": 3.05, "availability": "abundant"},
        "2025-11": {"avg": 2.24, "min": 1.81, "max": 3.02, "availability": "abundant"},
        "2025-12": {"avg": 2.22, "min": 1.80, "max": 3.00, "availability": "abundant"},
        "2026-01": {"avg": 2.20, "min": 1.78, "max": 2.97, "availability": "abundant"},
        "2026-02": {"avg": 2.18, "min": 1.75, "max": 2.95, "availability": "abundant"}
    },
    "H200": {
        "2024-04": {"avg": 5.20, "min": 4.70, "max": 6.20, "availability": "scarce"},
        "2024-05": {"avg": 5.05, "min": 4.55, "max": 6.05, "availability": "scarce"},
        "2024-06": {"avg": 4.90, "min": 4.40, "max": 5.90, "availability": "scarce"},
        "2024-07": {"avg": 4.70, "min": 4.10, "max": 5.70, "availability": "scarce"},
        "2024-08": {"avg": 4.50, "min": 3.90, "max": 5.50, "availability": "limited"},
        "2024-09": {"avg": 4.35, "min": 3.75, "max": 5.30, "availability": "limited"},
        "2024-10": {"avg": 4.15, "min": 3.55, "max": 5.00, "availability": "limited"},
        "2024-11": {"avg": 4.00, "min": 3.40, "max": 4.80, "availability": "limited"},
        "2024-12": {"avg": 3.85, "min": 3.30, "max": 4.60, "availability": "moderate"},
        "2025-01": {"avg": 3.70, "min": 3.20, "max": 4.45, "availability": "moderate"},
        "2025-02": {"avg": 3.60, "min": 3.12, "max": 4.35, "availability": "moderate"},
        "2025-03": {"avg": 3.52, "min": 3.05, "max": 4.25, "availability": "moderate"},
        "2025-04": {"avg": 3.45, "min": 2.98, "max": 4.15, "availability": "moderate"},
        "2025-05": {"avg": 3.40, "min": 2.92, "max": 4.08, "availability": "moderate"},
        "2025-06": {"avg": 3.35, "min": 2.88, "max": 4.00, "availability": "good"},
        "2025-07": {"avg": 3.30, "min": 2.84, "max": 3.92, "availability": "good"},
        "2025-08": {"avg": 3.25, "min": 2.80, "max": 3.85, "availability": "good"},
        "2025-09": {"avg": 3.22, "min": 2.78, "max": 3.80, "availability": "good"},
        "2025-10": {"avg": 3.18, "min": 2.74, "max": 3.75, "availability": "good"},
        "2025-11": {"avg": 3.15, "min": 2.70, "max": 3.70, "availability": "good"},
        "2025-12": {"avg": 3.12, "min": 2.68, "max": 3.66, "availability": "good"},
        "2026-01": {"avg": 3.08, "min": 2.65, "max": 3.60, "availability": "good"},
        "2026-02": {"avg": 3.05, "min": 2.60, "max": 3.55, "availability": "good"}
    },
    "A100-80GB": {
        "2022-01": {"avg": 3.80, "min": 3.20, "max": 4.50, "availability": "scarce"},
        "2022-04": {"avg": 3.50, "min": 2.90, "max": 4.20, "availability": "limited"},
        "2022-07": {"avg": 3.20, "min": 2.60, "max": 3.90, "availability": "limited"},
        "2022-10": {"avg": 3.00, "min": 2.40, "max": 3.60, "availability": "moderate"},
        "2023-01": {"avg": 2.80, "min": 2.20, "max": 3.40, "availability": "moderate"},
        "2023-02": {"avg": 2.72, "min": 2.15, "max": 3.35, "availability": "moderate"},
        "2023-03": {"avg": 2.65, "min": 2.08, "max": 3.28, "availability": "moderate"},
        "2023-04": {"avg": 2.55, "min": 2.00, "max": 3.18, "availability": "good"},
        "2023-05": {"avg": 2.48, "min": 1.95, "max": 3.10, "availability": "good"},
        "2023-06": {"avg": 2.40, "min": 1.88, "max": 3.00, "availability": "good"},
        "2023-07": {"avg": 2.30, "min": 1.78, "max": 2.90, "availability": "good"},
        "2023-08": {"avg": 2.22, "min": 1.72, "max": 2.82, "availability": "good"},
        "2023-09": {"avg": 2.15, "min": 1.68, "max": 2.75, "availability": "good"},
        "2023-10": {"avg": 2.08, "min": 1.60, "max": 2.68, "availability": "good"},
        "2023-11": {"avg": 2.02, "min": 1.55, "max": 2.62, "availability": "good"},
        "2023-12": {"avg": 1.95, "min": 1.48, "max": 2.55, "availability": "abundant"},
        "2024-01": {"avg": 1.88, "min": 1.40, "max": 2.48, "availability": "abundant"},
        "2024-02": {"avg": 1.82, "min": 1.35, "max": 2.42, "availability": "abundant"},
        "2024-03": {"avg": 1.75, "min": 1.30, "max": 2.35, "availability": "abundant"},
        "2024-04": {"avg": 1.68, "min": 1.25, "max": 2.28, "availability": "abundant"},
        "2024-05": {"avg": 1.62, "min": 1.22, "max": 2.22, "availability": "abundant"},
        "2024-06": {"avg": 1.57, "min": 1.18, "max": 2.15, "availability": "abundant"},
        "2024-07": {"avg": 1.52, "min": 1.14, "max": 2.12, "availability": "abundant"},
        "2024-08": {"avg": 1.48, "min": 1.10, "max": 2.08, "availability": "abundant"},
        "2024-09": {"avg": 1.45, "min": 1.08, "max": 2.05, "availability": "abundant"},
        "2024-10": {"avg": 1.42, "min": 1.06, "max": 2.02, "availability": "abundant"},
        "2024-11": {"avg": 1.39, "min": 1.04, "max": 1.98, "availability": "abundant"},
        "2024-12": {"avg": 1.36, "min": 1.02, "max": 1.95, "availability": "abundant"},
        "2025-01": {"avg": 1.34, "min": 1.00, "max": 1.92, "availability": "abundant"},
        "2025-02": {"avg": 1.32, "min": 0.98, "max": 1.90, "availability": "abundant"},
        "2025-03": {"avg": 1.30, "min": 0.97, "max": 1.88, "availability": "abundant"},
        "2025-04": {"avg": 1.29, "min": 0.96, "max": 1.86, "availability": "abundant"},
        "2025-05": {"avg": 1.28, "min": 0.95, "max": 1.85, "availability": "abundant"},
        "2025-06": {"avg": 1.27, "min": 0.94, "max": 1.84, "availability": "abundant"},
        "2025-07": {"avg": 1.27, "min": 0.93, "max": 1.82, "availability": "abundant"},
        "2025-08": {"avg": 1.26, "min": 0.92, "max": 1.80, "availability": "abundant"},
        "2025-09": {"avg": 1.26, "min": 0.91, "max": 1.79, "availability": "abundant"},
        "2025-10": {"avg": 1.25, "min": 0.90, "max": 1.78, "availability": "abundant"},
        "2025-11": {"avg": 1.24, "min": 0.89, "max": 1.76, "availability": "abundant"},
        "2025-12": {"avg": 1.24, "min": 0.88, "max": 1.75, "availability": "abundant"},
        "2026-01": {"avg": 1.23, "min": 0.87, "max": 1.73, "availability": "abundant"},
        "2026-02": {"avg": 1.22, "min": 0.85, "max": 1.70, "availability": "abundant"}
    },
    "A100-40GB": {
        "2022-01": {"avg": 2.50, "min": 2.00, "max": 3.00, "availability": "limited"},
        "2022-04": {"avg": 2.30, "min": 1.80, "max": 2.80, "availability": "moderate"},
        "2022-07": {"avg": 2.10, "min": 1.60, "max": 2.60, "availability": "moderate"},
        "2022-10": {"avg": 1.90, "min": 1.40, "max": 2.40, "availability": "good"},
        "2023-01": {"avg": 1.70, "min": 1.20, "max": 2.20, "availability": "good"},
        "2023-04": {"avg": 1.50, "min": 1.10, "max": 2.00, "availability": "good"},
        "2023-07": {"avg": 1.30, "min": 0.90, "max": 1.80, "availability": "abundant"},
        "2023-10": {"avg": 1.15, "min": 0.80, "max": 1.60, "availability": "abundant"},
        "2024-01": {"avg": 1.05, "min": 0.70, "max": 1.45, "availability": "abundant"},
        "2024-04": {"avg": 0.95, "min": 0.65, "max": 1.35, "availability": "abundant"},
        "2024-07": {"avg": 0.90, "min": 0.60, "max": 1.25, "availability": "abundant"},
        "2024-10": {"avg": 0.85, "min": 0.58, "max": 1.18, "availability": "abundant"},
        "2025-01": {"avg": 0.82, "min": 0.55, "max": 1.15, "availability": "abundant"},
        "2025-04": {"avg": 0.80, "min": 0.53, "max": 1.12, "availability": "abundant"},
        "2025-07": {"avg": 0.78, "min": 0.52, "max": 1.10, "availability": "abundant"},
        "2025-10": {"avg": 0.77, "min": 0.50, "max": 1.08, "availability": "abundant"},
        "2026-01": {"avg": 0.76, "min": 0.49, "max": 1.06, "availability": "abundant"},
        "2026-02": {"avg": 0.75, "min": 0.48, "max": 1.05, "availability": "abundant"}
    },
    "MI300X": {
        "2024-01": {"avg": 3.60, "min": 3.10, "max": 4.30, "availability": "scarce"},
        "2024-02": {"avg": 3.50, "min": 3.00, "max": 4.20, "availability": "scarce"},
        "2024-03": {"avg": 3.40, "min": 2.92, "max": 4.10, "availability": "scarce"},
        "2024-04": {"avg": 3.30, "min": 2.82, "max": 3.95, "availability": "limited"},
        "2024-05": {"avg": 3.20, "min": 2.72, "max": 3.82, "availability": "limited"},
        "2024-06": {"avg": 3.10, "min": 2.65, "max": 3.70, "availability": "limited"},
        "2024-07": {"avg": 3.00, "min": 2.55, "max": 3.58, "availability": "limited"},
        "2024-08": {"avg": 2.90, "min": 2.45, "max": 3.48, "availability": "moderate"},
        "2024-09": {"avg": 2.82, "min": 2.38, "max": 3.40, "availability": "moderate"},
        "2024-10": {"avg": 2.72, "min": 2.28, "max": 3.30, "availability": "moderate"},
        "2024-11": {"avg": 2.62, "min": 2.18, "max": 3.20, "availability": "moderate"},
        "2024-12": {"avg": 2.52, "min": 2.08, "max": 3.10, "availability": "moderate"},
        "2025-01": {"avg": 2.45, "min": 2.00, "max": 3.02, "availability": "good"},
        "2025-02": {"avg": 2.40, "min": 1.95, "max": 2.98, "availability": "good"},
        "2025-03": {"avg": 2.35, "min": 1.90, "max": 2.92, "availability": "good"},
        "2025-04": {"avg": 2.30, "min": 1.86, "max": 2.86, "availability": "good"},
        "2025-05": {"avg": 2.25, "min": 1.82, "max": 2.80, "availability": "good"},
        "2025-06": {"avg": 2.22, "min": 1.80, "max": 2.76, "availability": "good"},
        "2025-07": {"avg": 2.18, "min": 1.76, "max": 2.70, "availability": "good"},
        "2025-08": {"avg": 2.14, "min": 1.72, "max": 2.65, "availability": "good"},
        "2025-09": {"avg": 2.10, "min": 1.70, "max": 2.60, "availability": "good"},
        "2025-10": {"avg": 2.06, "min": 1.66, "max": 2.56, "availability": "good"},
        "2025-11": {"avg": 2.02, "min": 1.62, "max": 2.52, "availability": "good"},
        "2025-12": {"avg": 1.98, "min": 1.58, "max": 2.48, "availability": "good"},
        "2026-01": {"avg": 1.95, "min": 1.55, "max": 2.45, "availability": "good"},
        "2026-02": {"avg": 1.90, "min": 1.50, "max": 2.40, "availability": "good"}
    },
    "MI325X": {
        "2025-03": {"avg": 4.80, "min": 4.20, "max": 5.60, "availability": "scarce"},
        "2025-04": {"avg": 4.60, "min": 4.00, "max": 5.40, "availability": "scarce"},
        "2025-05": {"avg": 4.40, "min": 3.80, "max": 5.20, "availability": "scarce"},
        "2025-06": {"avg": 4.25, "min": 3.65, "max": 5.05, "availability": "limited"},
        "2025-07": {"avg": 4.10, "min": 3.52, "max": 4.90, "availability": "limited"},
        "2025-08": {"avg": 3.95, "min": 3.40, "max": 4.75, "availability": "limited"},
        "2025-09": {"avg": 3.82, "min": 3.28, "max": 4.60, "availability": "limited"},
        "2025-10": {"avg": 3.70, "min": 3.18, "max": 4.48, "availability": "moderate"},
        "2025-11": {"avg": 3.60, "min": 3.08, "max": 4.38, "availability": "moderate"},
        "2025-12": {"avg": 3.50, "min": 3.00, "max": 4.28, "availability": "moderate"},
        "2026-01": {"avg": 3.40, "min": 2.92, "max": 4.18, "availability": "moderate"},
        "2026-02": {"avg": 3.32, "min": 2.85, "max": 4.10, "availability": "moderate"}
    },
    "MI250X": {
        "2022-07": {"avg": 2.80, "min": 2.20, "max": 3.50, "availability": "limited"},
        "2022-10": {"avg": 2.60, "min": 2.00, "max": 3.30, "availability": "moderate"},
        "2023-01": {"avg": 2.40, "min": 1.80, "max": 3.10, "availability": "moderate"},
        "2023-04": {"avg": 2.20, "min": 1.65, "max": 2.85, "availability": "good"},
        "2023-07": {"avg": 2.00, "min": 1.50, "max": 2.60, "availability": "good"},
        "2023-10": {"avg": 1.80, "min": 1.35, "max": 2.40, "availability": "good"},
        "2024-01": {"avg": 1.60, "min": 1.20, "max": 2.15, "availability": "abundant"},
        "2024-04": {"avg": 1.40, "min": 1.05, "max": 1.90, "availability": "abundant"},
        "2024-07": {"avg": 1.25, "min": 0.95, "max": 1.70, "availability": "abundant"},
        "2024-10": {"avg": 1.12, "min": 0.85, "max": 1.55, "availability": "abundant"},
        "2025-01": {"avg": 1.00, "min": 0.78, "max": 1.40, "availability": "abundant"},
        "2025-04": {"avg": 0.92, "min": 0.72, "max": 1.30, "availability": "abundant"},
        "2025-07": {"avg": 0.88, "min": 0.68, "max": 1.22, "availability": "abundant"},
        "2025-10": {"avg": 0.85, "min": 0.65, "max": 1.18, "availability": "abundant"},
        "2026-01": {"avg": 0.82, "min": 0.62, "max": 1.15, "availability": "abundant"},
        "2026-02": {"avg": 0.80, "min": 0.60, "max": 1.12, "availability": "abundant"}
    },
    "RTX-4090": {
        "2023-01": {"avg": 0.82, "min": 0.52, "max": 1.22, "availability": "moderate"},
        "2023-04": {"avg": 0.76, "min": 0.46, "max": 1.12, "availability": "good"},
        "2023-07": {"avg": 0.70, "min": 0.42, "max": 1.02, "availability": "good"},
        "2023-10": {"avg": 0.65, "min": 0.38, "max": 0.95, "availability": "good"},
        "2024-01": {"avg": 0.60, "min": 0.34, "max": 0.90, "availability": "abundant"},
        "2024-04": {"avg": 0.55, "min": 0.30, "max": 0.85, "availability": "abundant"},
        "2024-07": {"avg": 0.50, "min": 0.28, "max": 0.80, "availability": "abundant"},
        "2024-10": {"avg": 0.46, "min": 0.26, "max": 0.75, "availability": "abundant"},
        "2025-01": {"avg": 0.43, "min": 0.24, "max": 0.72, "availability": "abundant"},
        "2025-04": {"avg": 0.40, "min": 0.22, "max": 0.68, "availability": "abundant"},
        "2025-07": {"avg": 0.38, "min": 0.21, "max": 0.66, "availability": "abundant"},
        "2025-10": {"avg": 0.37, "min": 0.20, "max": 0.64, "availability": "abundant"},
        "2026-01": {"avg": 0.36, "min": 0.19, "max": 0.62, "availability": "abundant"},
        "2026-02": {"avg": 0.35, "min": 0.18, "max": 0.60, "availability": "abundant"}
    },
    "L40S": {
        "2023-10": {"avg": 1.30, "min": 1.05, "max": 1.60, "availability": "moderate"},
        "2024-01": {"avg": 1.22, "min": 0.98, "max": 1.50, "availability": "good"},
        "2024-04": {"avg": 1.15, "min": 0.92, "max": 1.42, "availability": "good"},
        "2024-07": {"avg": 1.08, "min": 0.86, "max": 1.35, "availability": "abundant"},
        "2024-10": {"avg": 1.02, "min": 0.82, "max": 1.28, "availability": "abundant"},
        "2025-01": {"avg": 0.98, "min": 0.78, "max": 1.22, "availability": "abundant"},
        "2025-04": {"avg": 0.95, "min": 0.75, "max": 1.18, "availability": "abundant"},
        "2025-07": {"avg": 0.92, "min": 0.73, "max": 1.15, "availability": "abundant"},
        "2025-10": {"avg": 0.90, "min": 0.72, "max": 1.13, "availability": "abundant"},
        "2026-01": {"avg": 0.88, "min": 0.70, "max": 1.10, "availability": "abundant"},
        "2026-02": {"avg": 0.87, "min": 0.69, "max": 1.08, "availability": "abundant"}
    }
}

# ============================================================================
# MARKET DATA & DEMAND INDICATORS
# ============================================================================

MARKET_INDICATORS = {
    "nvidia_stock": {"ticker": "NVDA", "current": 138.25, "1m_change": 4.2, "3m_change": 12.8, "ytd_change": 8.5, "52w_high": 153.13, "52w_low": 75.61},
    "amd_stock": {"ticker": "AMD", "current": 119.50, "1m_change": -2.1, "3m_change": 5.3, "ytd_change": 3.8, "52w_high": 164.46, "52w_low": 100.55},
    "gpu_market_size_bn": {"2023": 52.4, "2024": 71.2, "2025": 95.8, "2026_est": 128.5, "2027_est": 168.3},
    "ai_capex_bn": {"2023": 55, "2024": 95, "2025_est": 150, "2026_est": 210, "2027_est": 280},
    "data_center_gpu_shipments_k": {"2023-Q1": 480, "2023-Q2": 520, "2023-Q3": 610, "2023-Q4": 750,
                                     "2024-Q1": 850, "2024-Q2": 920, "2024-Q3": 1050, "2024-Q4": 1180,
                                     "2025-Q1": 1320, "2025-Q2": 1450, "2025-Q3": 1580, "2025-Q4": 1700,
                                     "2026-Q1": 1850},
    "h100_lead_time_weeks": {"2023-01": 52, "2023-03": 46, "2023-06": 40, "2023-09": 36, "2023-12": 28,
                              "2024-03": 20, "2024-06": 16, "2024-09": 12, "2024-12": 8,
                              "2025-03": 6, "2025-06": 4, "2025-09": 3, "2025-12": 2, "2026-01": 1, "2026-02": 1},
    "amd_gpu_market_share_pct": {"2023-01": 3, "2023-06": 5, "2024-01": 8, "2024-06": 12, "2025-01": 16, "2025-06": 19, "2026-01": 22}
}

# ============================================================================
# REGIONAL ADOPTION & DEMAND (with per-GPU pricing)
# ============================================================================

REGIONAL_DATA = {
    "North America": {
        "market_share_pct": 42.5,
        "yoy_growth_pct": 28.3,
        "top_providers": ["AWS", "GCP", "Azure", "CoreWeave", "Lambda"],
        "gpu_demand_index": 95,
        "key_hubs": ["Virginia", "Oregon", "Texas", "California"],
        "avg_price_premium_pct": 0,
        "energy_cost_kwh": 0.065,
        "regulatory_score": 8.5,
        "data_centers_count": 1850,
        "gpu_pricing": {
            "H100-SXM": {"avg": 2.80, "low": 2.15, "high": 4.28},
            "B200": {"avg": 4.50, "low": 3.75, "high": 5.35},
            "A100-80GB": {"avg": 1.55, "low": 1.10, "high": 2.52},
"MI300X": {"avg": 2.30, "low": 1.85, "high": 3.15}
        },
        "adoption_trend_monthly": {
            "2024-01": 38.0, "2024-04": 39.2, "2024-07": 40.1, "2024-10": 41.0,
            "2025-01": 41.5, "2025-04": 41.8, "2025-07": 42.0, "2025-10": 42.3,
            "2026-01": 42.5, "2026-02": 42.5
        }
    },
    "Europe": {
        "market_share_pct": 22.8,
        "yoy_growth_pct": 32.1,
        "top_providers": ["Azure", "GCP", "AWS", "OVH"],
        "gpu_demand_index": 78,
        "key_hubs": ["Frankfurt", "Dublin", "Amsterdam", "London", "Paris"],
        "avg_price_premium_pct": 10,
        "energy_cost_kwh": 0.12,
        "regulatory_score": 7.0,
        "data_centers_count": 920,
        "gpu_pricing": {
            "H100-SXM": {"avg": 3.50, "low": 2.35, "high": 4.73},
            "B200": {"avg": 5.40, "low": 4.68, "high": 5.89},
            "A100-80GB": {"avg": 1.85, "low": 1.25, "high": 2.84},
"MI300X": {"avg": 2.75, "low": 2.05, "high": 3.47}
        },
        "adoption_trend_monthly": {
            "2024-01": 19.5, "2024-04": 20.2, "2024-07": 20.8, "2024-10": 21.4,
            "2025-01": 21.8, "2025-04": 22.0, "2025-07": 22.3, "2025-10": 22.5,
            "2026-01": 22.8, "2026-02": 22.8
        }
    },
    "Asia Pacific": {
        "market_share_pct": 24.3,
        "yoy_growth_pct": 38.7,
        "top_providers": ["AWS", "GCP", "Azure", "Alibaba", "Tencent"],
        "gpu_demand_index": 88,
        "key_hubs": ["Tokyo", "Singapore", "Mumbai", "Sydney", "Seoul"],
        "avg_price_premium_pct": 15,
        "energy_cost_kwh": 0.09,
        "regulatory_score": 6.5,
        "data_centers_count": 780,
        "gpu_pricing": {
            "H100-SXM": {"avg": 3.95, "low": 2.50, "high": 5.14},
            "B200": {"avg": 5.90, "low": 5.58, "high": 6.42},
            "A100-80GB": {"avg": 2.10, "low": 1.35, "high": 3.03},
"MI300X": {"avg": 2.95, "low": 2.40, "high": 3.68}
        },
        "adoption_trend_monthly": {
            "2024-01": 20.5, "2024-04": 21.2, "2024-07": 21.8, "2024-10": 22.5,
            "2025-01": 23.0, "2025-04": 23.4, "2025-07": 23.7, "2025-10": 24.0,
            "2026-01": 24.3, "2026-02": 24.5
        }
    },
    "Middle East & Africa": {
        "market_share_pct": 4.2,
        "yoy_growth_pct": 52.4,
        "top_providers": ["Azure", "AWS", "Oracle", "G42"],
        "gpu_demand_index": 45,
        "key_hubs": ["UAE", "Saudi Arabia", "South Africa"],
        "avg_price_premium_pct": 20,
        "energy_cost_kwh": 0.04,
        "regulatory_score": 5.0,
        "data_centers_count": 120,
        "gpu_pricing": {
            "H100-SXM": {"avg": 4.50, "low": 3.80, "high": 5.50},
            "A100-80GB": {"avg": 2.40, "low": 1.80, "high": 3.20}
        },
        "adoption_trend_monthly": {
            "2024-01": 2.5, "2024-04": 2.8, "2024-07": 3.1, "2024-10": 3.4,
            "2025-01": 3.6, "2025-04": 3.8, "2025-07": 4.0, "2025-10": 4.1,
            "2026-01": 4.2, "2026-02": 4.3
        }
    },
    "Latin America": {
        "market_share_pct": 3.8,
        "yoy_growth_pct": 44.2,
        "top_providers": ["AWS", "Azure", "GCP", "Oracle"],
        "gpu_demand_index": 35,
        "key_hubs": ["Sao Paulo", "Mexico City", "Santiago"],
        "avg_price_premium_pct": 18,
        "energy_cost_kwh": 0.08,
        "regulatory_score": 5.5,
        "data_centers_count": 95,
        "gpu_pricing": {
            "H100-SXM": {"avg": 4.30, "low": 3.60, "high": 5.20},
            "A100-80GB": {"avg": 2.25, "low": 1.65, "high": 3.00}
        },
        "adoption_trend_monthly": {
            "2024-01": 2.2, "2024-04": 2.5, "2024-07": 2.8, "2024-10": 3.1,
            "2025-01": 3.3, "2025-04": 3.5, "2025-07": 3.6, "2025-10": 3.7,
            "2026-01": 3.8, "2026-02": 3.9
        }
    },
    "China (Domestic)": {
        "market_share_pct": 2.4,
        "yoy_growth_pct": 15.8,
        "top_providers": ["Alibaba", "Tencent", "Huawei", "Baidu"],
        "gpu_demand_index": 72,
        "key_hubs": ["Beijing", "Shanghai", "Shenzhen", "Guizhou"],
        "avg_price_premium_pct": 25,
        "energy_cost_kwh": 0.06,
        "regulatory_score": 4.0,
        "data_centers_count": 450,
        "gpu_pricing": {
            "A100-80GB": {"avg": 2.80, "low": 2.20, "high": 3.60},
            "A100-40GB": {"avg": 2.10, "low": 1.60, "high": 2.80}
        },
        "adoption_trend_monthly": {
            "2024-01": 2.8, "2024-04": 2.7, "2024-07": 2.6, "2024-10": 2.5,
            "2025-01": 2.5, "2025-04": 2.4, "2025-07": 2.4, "2025-10": 2.4,
            "2026-01": 2.4, "2026-02": 2.4
        }
    }
}

# ============================================================================
# WORKLOAD RECOMMENDATIONS
# ============================================================================

WORKLOAD_RECOMMENDATIONS = {
    "LLM Training (>70B params)": {
        "recommended": ["B200", "H100-SXM", "H200", "GB200"],
        "min_gpus": 64,
        "budget_monthly_low": 150000,
        "budget_monthly_high": 2000000,
        "best_value": "Lambda B200 cluster"
    },
    "LLM Training (7B-70B)": {
        "recommended": ["H100-SXM", "A100-80GB", "MI300X", "B200"],
        "min_gpus": 8,
        "budget_monthly_low": 15000,
        "budget_monthly_high": 200000,
        "best_value": "CoreWeave H100"
    },
    "LLM Fine-tuning": {
        "recommended": ["A100-80GB", "MI300X", "H100-SXM", "L40S"],
        "min_gpus": 1,
        "budget_monthly_low": 1000,
        "budget_monthly_high": 25000,
        "best_value": "Vast.ai MI300X"
    },
    "LLM Inference": {
        "recommended": ["A10G", "L40S"],
        "min_gpus": 1,
        "budget_monthly_low": 300,
        "budget_monthly_high": 5000,
        "best_value": "GCP A10G"
    },
    "Image/Video Generation": {
        "recommended": ["RTX-4090", "L40S", "A100-40GB", "MI250X"],
        "min_gpus": 1,
        "budget_monthly_low": 200,
        "budget_monthly_high": 3000,
        "best_value": "Vast.ai RTX 4090"
    },
    "Research / Experimentation": {
        "recommended": ["A10G", "RTX-4090", "MI210"],
        "min_gpus": 1,
        "budget_monthly_low": 50,
        "budget_monthly_high": 1000,
        "best_value": "Vast.ai RTX-4090 spot"
    }
}

# ============================================================================
# TCO (TOTAL COST OF OWNERSHIP) MODEL
# ============================================================================

TCO_COMPONENTS = {
    "networking": {
        "description": "Inter-node networking (InfiniBand/RoCE)",
        "cost_per_gpu_hr": {
            "H100-SXM": 0.35, "B200": 0.45, "H200": 0.38, "GB200": 0.55,
            "A100-80GB": 0.25, "A100-40GB": 0.20, "MI300X": 0.30, "MI325X": 0.35,
            "L40S": 0.12, "A10G": 0.05,
            "RTX-4090": 0.08, "RTX-5090": 0.10, "MI250X": 0.22
        }
    },
    "storage": {
        "description": "NVMe/SSD storage (per GPU allocation)",
        "cost_per_gpu_hr": {
            "H100-SXM": 0.18, "B200": 0.22, "H200": 0.20, "GB200": 0.30,
            "A100-80GB": 0.15, "A100-40GB": 0.12, "MI300X": 0.16, "MI325X": 0.18,
            "L40S": 0.08, "A10G": 0.04,
            "RTX-4090": 0.05, "RTX-5090": 0.06, "MI250X": 0.12
        }
    },
    "egress": {
        "description": "Data egress (avg per GPU-hr at typical workloads)",
        "cost_per_gpu_hr": {
            "H100-SXM": 0.08, "B200": 0.10, "H200": 0.09, "GB200": 0.12,
            "A100-80GB": 0.06, "A100-40GB": 0.05, "MI300X": 0.07, "MI325X": 0.08,
            "L40S": 0.04, "A10G": 0.02,
            "RTX-4090": 0.03, "RTX-5090": 0.03, "MI250X": 0.05
        }
    },
    "energy_overhead": {
        "description": "Energy + cooling beyond GPU TDP (PUE ~1.3)",
        "cost_per_gpu_hr": {
            "H100-SXM": 0.14, "B200": 0.20, "H200": 0.14, "GB200": 0.42,
            "A100-80GB": 0.08, "A100-40GB": 0.08, "MI300X": 0.15, "MI325X": 0.16,
            "L40S": 0.07, "A10G": 0.03,
            "RTX-4090": 0.09, "RTX-5090": 0.12, "MI250X": 0.11
        }
    },
    "ops_management": {
        "description": "Platform/management overhead per GPU-hr",
        "cost_per_gpu_hr": {
            "H100-SXM": 0.10, "B200": 0.12, "H200": 0.10, "GB200": 0.15,
            "A100-80GB": 0.08, "A100-40GB": 0.06, "MI300X": 0.10, "MI325X": 0.10,
            "L40S": 0.05, "A10G": 0.03,
            "RTX-4090": 0.04, "RTX-5090": 0.05, "MI250X": 0.06
        }
    }
}

# ============================================================================
# INFERENCE ECONOMICS — $/M tokens by model and GPU
# ============================================================================

INFERENCE_BENCHMARKS = {
    # ── Top 20 models by usage on OpenRouter (Feb 2026) ──
    # Pricing: $/M tokens from OpenRouter API + major providers
    # Ranked by real usage volume from millions of users

    # #1 — Grok Code Fast (xAI)
    "Grok-Code-Fast": {"params_b": 70, "type": "Code", "category": "Large", "rank": 1,
        "gpus": {"H100-SXM": {"tokens_per_sec": 105, "cost_per_1m_tokens": 0.35, "vram_gb": 42}, "B200": {"tokens_per_sec": 200, "cost_per_1m_tokens": 0.20, "vram_gb": 42}, "A100-80GB": {"tokens_per_sec": 58, "cost_per_1m_tokens": 0.65, "vram_gb": 42}},
        "providers": {"xAI API": 0.20, "OpenRouter": 0.20, "Together": 0.28}
    },
    # #2 — Grok 4 Fast (xAI)
    "Grok-4-Fast": {"params_b": 314, "type": "LLM", "category": "Frontier", "rank": 2,
        "gpus": {"H100-SXM": {"tokens_per_sec": 22, "cost_per_1m_tokens": 2.80, "vram_gb": 280}, "B200": {"tokens_per_sec": 45, "cost_per_1m_tokens": 1.55, "vram_gb": 280}},
        "providers": {"xAI API": 0.20, "OpenRouter": 0.20}
    },
    # #3 — Claude Sonnet 4 (Anthropic)
    "Claude-Sonnet-4": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 3,
        "gpus": {},
        "providers": {"Anthropic API": 3.00, "OpenRouter": 3.00, "AWS Bedrock": 3.00, "Google Vertex": 3.00, "Azure": 3.00}
    },
    # #4 — Gemini 2.5 Flash (Google)
    "Gemini-2.5-Flash": {"params_b": 65, "type": "LLM", "category": "Large", "rank": 4,
        "gpus": {},
        "providers": {"Google AI Studio": 0.15, "Google Vertex": 0.30, "OpenRouter": 0.30}
    },
    # #5 — Claude Sonnet 4.5 (Anthropic)
    "Claude-Sonnet-4.5": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 5,
        "gpus": {},
        "providers": {"Anthropic API": 3.00, "OpenRouter": 3.00, "AWS Bedrock": 3.00, "Google Vertex": 3.00, "Azure": 3.00}
    },
    # #6 — DeepSeek V3.1 (DeepSeek)
    "DeepSeek-V3.1": {"params_b": 671, "type": "LLM", "category": "Frontier", "rank": 6,
        "gpus": {"H100-SXM": {"tokens_per_sec": 38, "cost_per_1m_tokens": 1.75, "vram_gb": 180}, "B200": {"tokens_per_sec": 78, "cost_per_1m_tokens": 0.90, "vram_gb": 180}},
        "providers": {"DeepSeek API": 0.15, "OpenRouter": 0.15, "Together": 0.75, "Fireworks": 0.80, "DeepInfra": 0.72}
    },
    # #7 — GPT-4.1 Mini (OpenAI)
    "GPT-4.1-Mini": {"params_b": 70, "type": "LLM", "category": "Large", "rank": 7,
        "gpus": {},
        "providers": {"OpenAI API": 0.40, "OpenRouter": 0.40, "Azure": 0.40}
    },
    # #8 — Gemini 2.0 Flash (Google)
    "Gemini-2.0-Flash": {"params_b": 50, "type": "LLM", "category": "Large", "rank": 8,
        "gpus": {},
        "providers": {"Google AI Studio": 0.10, "Google Vertex": 0.10, "OpenRouter": 0.10}
    },
    # #9 — Gemini 2.5 Flash Lite (Google)
    "Gemini-2.5-Flash-Lite": {"params_b": 30, "type": "LLM", "category": "Medium", "rank": 9,
        "gpus": {},
        "providers": {"Google AI Studio": 0.05, "Google Vertex": 0.10, "OpenRouter": 0.10}
    },
    # #10 — DeepSeek V3 (DeepSeek)
    "DeepSeek-V3": {"params_b": 671, "type": "LLM", "category": "Frontier", "rank": 10,
        "gpus": {"H100-SXM": {"tokens_per_sec": 35, "cost_per_1m_tokens": 1.85, "vram_gb": 180}, "B200": {"tokens_per_sec": 72, "cost_per_1m_tokens": 0.95, "vram_gb": 180}},
        "providers": {"DeepSeek API": 0.19, "OpenRouter": 0.19, "Together": 0.80, "Fireworks": 0.87, "DeepInfra": 0.78}
    },
    # #11 — GPT-5 (OpenAI)
    "GPT-5": {"params_b": 500, "type": "LLM", "category": "Frontier", "rank": 11,
        "gpus": {},
        "providers": {"OpenAI API": 1.75, "OpenRouter": 1.75, "Azure": 1.75}
    },
    # #12 — Gemini 2.5 Pro (Google)
    "Gemini-2.5-Pro": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 12,
        "gpus": {},
        "providers": {"Google AI Studio": 1.25, "Google Vertex": 1.25, "OpenRouter": 1.25}
    },
    # #13 — Qwen3 30B-A3B (Alibaba)
    "Qwen3-30B-A3B": {"params_b": 30, "type": "LLM", "category": "Medium", "rank": 13,
        "gpus": {"H100-SXM": {"tokens_per_sec": 320, "cost_per_1m_tokens": 0.10, "vram_gb": 5}, "B200": {"tokens_per_sec": 580, "cost_per_1m_tokens": 0.058, "vram_gb": 5}, "A100-80GB": {"tokens_per_sec": 180, "cost_per_1m_tokens": 0.18, "vram_gb": 5}, "MI300X": {"tokens_per_sec": 260, "cost_per_1m_tokens": 0.12, "vram_gb": 5}},
        "providers": {"OpenRouter": 0.06, "Together": 0.08, "Fireworks": 0.08, "DeepInfra": 0.05, "Groq": 0.05}
    },
    # #14 — Llama 3.3 70B (Meta)
    "Llama-3.3-70B": {"params_b": 70, "type": "LLM", "category": "Large", "rank": 14,
        "gpus": {"H100-SXM": {"tokens_per_sec": 100, "cost_per_1m_tokens": 0.36, "vram_gb": 42}, "B200": {"tokens_per_sec": 195, "cost_per_1m_tokens": 0.21, "vram_gb": 42}, "H200": {"tokens_per_sec": 125, "cost_per_1m_tokens": 0.30, "vram_gb": 42}, "A100-80GB": {"tokens_per_sec": 55, "cost_per_1m_tokens": 0.68, "vram_gb": 42}, "MI300X": {"tokens_per_sec": 82, "cost_per_1m_tokens": 0.45, "vram_gb": 42}},
        "providers": {"OpenRouter": 0.10, "Together": 0.27, "Fireworks": 0.28, "Groq": 0.10, "AWS Bedrock": 0.45, "Azure": 0.42, "DeepInfra": 0.28}
    },
    # #15 — DeepSeek R1 (DeepSeek)
    "DeepSeek-R1": {"params_b": 671, "type": "LLM", "category": "Frontier", "rank": 15,
        "gpus": {"H100-SXM": {"tokens_per_sec": 28, "cost_per_1m_tokens": 2.10, "vram_gb": 180}, "B200": {"tokens_per_sec": 58, "cost_per_1m_tokens": 1.15, "vram_gb": 180}},
        "providers": {"DeepSeek API": 0.40, "OpenRouter": 0.40, "Together": 1.50, "Fireworks": 1.60, "DeepInfra": 1.40}
    },
    # #16 — Qwen3 Coder 480B (Alibaba)
    "Qwen3-Coder-480B": {"params_b": 480, "type": "Code", "category": "Frontier", "rank": 16,
        "gpus": {"H100-SXM": {"tokens_per_sec": 15, "cost_per_1m_tokens": 3.50, "vram_gb": 350}, "B200": {"tokens_per_sec": 32, "cost_per_1m_tokens": 1.90, "vram_gb": 350}},
        "providers": {"OpenRouter": 0.12, "Together": 1.00, "DeepInfra": 0.75, "Fireworks": 1.10}
    },
    # #17 — Claude 3.7 Sonnet (Anthropic)
    "Claude-3.7-Sonnet": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 17,
        "gpus": {},
        "providers": {"Anthropic API": 3.00, "OpenRouter": 3.00, "AWS Bedrock": 3.00, "Google Vertex": 3.00, "Azure": 3.00}
    },
    # #18 — GPT-4o Mini (OpenAI)
    "GPT-4o-Mini": {"params_b": 70, "type": "LLM", "category": "Large", "rank": 18,
        "gpus": {},
        "providers": {"OpenAI API": 0.15, "OpenRouter": 0.15, "Azure": 0.15}
    },
    # #19 — Gemini 2.5 Pro Preview (Google)
    "Gemini-2.5-Pro-Preview": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 19,
        "gpus": {},
        "providers": {"Google AI Studio": 1.25, "Google Vertex": 1.25, "OpenRouter": 1.25}
    },
    # #20 — GPT-5.2 (OpenAI)
    "GPT-5.2": {"params_b": 500, "type": "LLM", "category": "Frontier", "rank": 20,
        "gpus": {},
        "providers": {"OpenAI API": 1.75, "OpenRouter": 1.75, "Azure": 1.75}
    }
}

# ============================================================================
# SPOT MARKET DATA (simulated live marketplace data)
# ============================================================================

SPOT_MARKET = {
    "H100-SXM": {
        "bid": 1.82, "ask": 2.05, "spread_pct": 11.2, "last_trade": 1.95,
        "24h_low": 1.70, "24h_high": 2.18, "24h_volume_gpu_hrs": 48500,
        "7d_avg": 1.92, "30d_avg": 2.01, "volatility_30d": 8.5,
        "available_gpus": 2840, "queue_depth": 125,
        "hourly_prices_24h": [1.88,1.85,1.82,1.80,1.78,1.82,1.85,1.90,1.95,2.02,2.08,2.12,2.15,2.18,2.12,2.08,2.05,2.00,1.98,1.95,1.92,1.90,1.88,1.85]
    },
    "B200": {
        "bid": 3.10, "ask": 3.55, "spread_pct": 12.7, "last_trade": 3.35,
        "24h_low": 2.95, "24h_high": 3.80, "24h_volume_gpu_hrs": 12200,
        "7d_avg": 3.30, "30d_avg": 3.48, "volatility_30d": 14.2,
        "available_gpus": 680, "queue_depth": 310,
        "hourly_prices_24h": [3.20,3.15,3.10,3.05,3.10,3.18,3.25,3.35,3.45,3.55,3.65,3.75,3.80,3.72,3.60,3.50,3.45,3.40,3.38,3.35,3.30,3.28,3.25,3.20]
    },
    "H200": {
        "bid": 2.55, "ask": 2.85, "spread_pct": 10.5, "last_trade": 2.72,
        "24h_low": 2.45, "24h_high": 3.00, "24h_volume_gpu_hrs": 18900,
        "7d_avg": 2.68, "30d_avg": 2.78, "volatility_30d": 9.8,
        "available_gpus": 1250, "queue_depth": 180,
        "hourly_prices_24h": [2.60,2.58,2.55,2.52,2.55,2.60,2.65,2.72,2.78,2.85,2.92,2.98,3.00,2.95,2.88,2.82,2.78,2.75,2.72,2.70,2.68,2.65,2.62,2.60]
    },
    "A100-80GB": {
        "bid": 0.85, "ask": 1.02, "spread_pct": 16.7, "last_trade": 0.95,
        "24h_low": 0.80, "24h_high": 1.15, "24h_volume_gpu_hrs": 92300,
        "7d_avg": 0.93, "30d_avg": 0.98, "volatility_30d": 12.3,
        "available_gpus": 8500, "queue_depth": 45,
        "hourly_prices_24h": [0.90,0.88,0.86,0.85,0.85,0.87,0.90,0.92,0.95,0.98,1.02,1.05,1.08,1.12,1.15,1.10,1.05,1.02,0.98,0.96,0.94,0.92,0.91,0.90]
    },
    "MI300X": {
        "bid": 1.48, "ask": 1.72, "spread_pct": 13.9, "last_trade": 1.62,
        "24h_low": 1.40, "24h_high": 1.85, "24h_volume_gpu_hrs": 15600,
        "7d_avg": 1.58, "30d_avg": 1.68, "volatility_30d": 11.5,
        "available_gpus": 1820, "queue_depth": 95,
        "hourly_prices_24h": [1.52,1.50,1.48,1.46,1.48,1.52,1.56,1.62,1.68,1.72,1.78,1.82,1.85,1.80,1.75,1.72,1.68,1.65,1.62,1.60,1.58,1.55,1.53,1.52]
    },
    "RTX-4090": {
        "bid": 0.18, "ask": 0.28, "spread_pct": 35.7, "last_trade": 0.24,
        "24h_low": 0.16, "24h_high": 0.35, "24h_volume_gpu_hrs": 145000,
        "7d_avg": 0.23, "30d_avg": 0.25, "volatility_30d": 22.5,
        "available_gpus": 15200, "queue_depth": 12,
        "hourly_prices_24h": [0.22,0.20,0.19,0.18,0.18,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.32,0.30,0.28,0.26,0.25,0.24,0.23,0.22,0.21,0.20]
    }
}

# ============================================================================
# NEWS & MARKET SIGNALS
# ============================================================================

NEWS_FEED = [
    {"date": "2026-02-17", "source": "Motley Fool", "headline": "NVIDIA Q4 FY2026 earnings due Feb 25 -- Amazon, Google, Meta, Microsoft capex plans boost NVDA outlook", "category": "earnings", "sentiment": "bullish", "impact": "high"},
    {"date": "2026-02-14", "source": "Motley Fool", "headline": "CoreWeave stock jumps 7% ahead of Feb 26 earnings; shares up 34% YTD to $96", "category": "market", "sentiment": "bullish", "impact": "medium"},
    {"date": "2026-02-12", "source": "CNBC", "headline": "Anthropic closes $30B Series G at $380B valuation -- annualized revenue hits $14B", "category": "demand", "sentiment": "bullish", "impact": "high"},
    {"date": "2026-02-12", "source": "TechCrunch", "headline": "OpenAI releases GPT-5.3-Codex-Spark; Google launches Gemini 3 Deep Think -- GPU demand intensifies", "category": "demand", "sentiment": "bullish", "impact": "high"},
    {"date": "2026-02-07", "source": "Wolf Street", "headline": "Big Tech plans $700B in AI capex for 2026: Amazon $200B, Google $185B, Microsoft $145B, Meta $135B", "category": "demand", "sentiment": "bullish", "impact": "high"},
    {"date": "2026-02-06", "source": "CNBC", "headline": "Tech AI spending approaches $700B in 2026 as free cash flow takes major hit across hyperscalers", "category": "demand", "sentiment": "neutral", "impact": "high"},
    {"date": "2026-02-06", "source": "Bloomberg", "headline": "NVIDIA H200 chip sales to China stalled by US national security review despite Trump approval", "category": "policy", "sentiment": "negative", "impact": "high"},
    {"date": "2026-02-04", "source": "CNBC", "headline": "NVIDIA AI chip sales to China remain in limbo -- State Dept pushes for tougher H200 export restrictions", "category": "policy", "sentiment": "negative", "impact": "high"},
    {"date": "2026-02-01", "source": "AI Business 2.0", "headline": "TSMC and ASML capacity constraints tighten -- CoWoS advanced packaging sold out through 2026", "category": "supply", "sentiment": "bearish", "impact": "high"},
    {"date": "2026-01-26", "source": "CNBC", "headline": "NVIDIA invests $2B in CoreWeave to accelerate AI factory buildout targeting 5GW capacity by 2030", "category": "expansion", "sentiment": "positive", "impact": "high"},
    {"date": "2026-01-15", "source": "Federal Register", "headline": "BIS revises AI chip export policy for China: H200 shifts from presumption of denial to case-by-case review", "category": "policy", "sentiment": "neutral", "impact": "high"},
    {"date": "2026-01-13", "source": "TechSpot", "headline": "GPU prices up 19% in 3 months -- DRAM crisis and AI demand drive RTX 5090 to 65% above MSRP", "category": "pricing", "sentiment": "bearish", "impact": "medium"},
    {"date": "2026-01-05", "source": "The Register", "headline": "AWS raises GPU instance prices 15% -- H200 p5e.48xlarge jumps from $34.61 to $39.80/hr", "category": "pricing", "sentiment": "bearish", "impact": "high"},
    {"date": "2026-01-05", "source": "TrendForce", "headline": "NVIDIA and AMD plan phased GPU price hikes in Q1 2026 as HBM memory costs surge", "category": "pricing", "sentiment": "bearish", "impact": "medium"},
    {"date": "2026-01-05", "source": "MIT Tech Review", "headline": "AI reasoning models drive insatiable GPU demand -- NVIDIA delays gaming GPUs as AI factories take priority", "category": "demand", "sentiment": "bullish", "impact": "medium"}
]

# ============================================================================
# DATA AGGREGATION FUNCTIONS
# ============================================================================

def get_cheapest_by_gpu(gpu_id: str) -> list:
    """Get all providers sorted by price for a specific GPU."""
    results = []
    for provider, data in CLOUD_PRICING.items():
        if gpu_id in data["gpus"]:
            gpu = data["gpus"][gpu_id]
            results.append({
                "provider": provider,
                "provider_name": data["provider_name"],
                "instance": gpu["instance"],
                "price_per_gpu_hr": gpu["price_per_gpu_hr"],
                "price_monthly": gpu["price_per_gpu_hr"] * 730,
                "spot_price": gpu["price_per_gpu_hr"] * (1 - data["spot_discount"]) if data["spot_discount"] > 0 else None,
                "reserved_1yr": gpu["price_per_gpu_hr"] * (1 - data["reserved_1yr_discount"]),
                "reserved_3yr": gpu["price_per_gpu_hr"] * (1 - data["reserved_3yr_discount"]),
                "regions": gpu.get("regions", {})
            })
    results.sort(key=lambda x: x["price_per_gpu_hr"])
    return results


def get_price_trends(gpu_id: str) -> dict:
    """Get historical price trends for a GPU."""
    if gpu_id in HISTORICAL_PRICING:
        return HISTORICAL_PRICING[gpu_id]
    return {}


def get_all_gpu_prices() -> dict:
    """Get a matrix of all GPU prices across providers."""
    matrix = {}
    for gpu_id in GPU_SPECS:
        matrix[gpu_id] = get_cheapest_by_gpu(gpu_id)
    return matrix


def get_price_comparison_matrix() -> list:
    """Create a comprehensive price comparison matrix."""
    rows = []
    for gpu_id, spec in GPU_SPECS.items():
        providers = get_cheapest_by_gpu(gpu_id)
        if not providers:
            continue
        cheapest = providers[0]["price_per_gpu_hr"]
        most_expensive = providers[-1]["price_per_gpu_hr"] if providers else cheapest
        avg_price = sum(p["price_per_gpu_hr"] for p in providers) / len(providers) if providers else 0

        # Get trend data — monthly change
        trends = get_price_trends(gpu_id)
        trend_periods = sorted(trends.keys())
        if len(trend_periods) >= 2:
            latest = trends[trend_periods[-1]]["avg"]
            prev = trends[trend_periods[-2]]["avg"]
            price_change_pct = ((latest - prev) / prev) * 100
        else:
            price_change_pct = 0

        rows.append({
            "gpu_id": gpu_id,
            "name": spec["name"],
            "vendor": spec.get("vendor", "NVIDIA"),
            "vram_gb": spec["vram_gb"],
            "arch": spec["arch"],
            "tier": spec["tier"],
            "cheapest_price": cheapest,
            "cheapest_provider": providers[0]["provider"] if providers else "N/A",
            "most_expensive": most_expensive,
            "avg_price": round(avg_price, 2),
            "num_providers": len(providers),
            "price_spread_pct": round(((most_expensive - cheapest) / cheapest) * 100, 1) if cheapest > 0 else 0,
            "monthly_change_pct": round(price_change_pct, 1),
            "flops_per_dollar": round(spec["fp16_tflops"] / cheapest, 1) if cheapest > 0 else 0,
            "vram_per_dollar": round(spec["vram_gb"] / cheapest, 1) if cheapest > 0 else 0
        })
    rows.sort(key=lambda x: x["cheapest_price"], reverse=True)
    return rows


def generate_market_summary() -> dict:
    """Generate a comprehensive market summary."""
    comparison = get_price_comparison_matrix()

    best_flops_per_dollar = max(comparison, key=lambda x: x["flops_per_dollar"])
    best_vram_per_dollar = max(comparison, key=lambda x: x["vram_per_dollar"])
    biggest_price_drop = min(comparison, key=lambda x: x["monthly_change_pct"])
    most_competitive = max(comparison, key=lambda x: x["num_providers"])

    return {
        "timestamp": datetime.now().isoformat(),
        "total_gpus_tracked": len(GPU_SPECS),
        "total_providers_tracked": len(CLOUD_PRICING),
        "best_flops_per_dollar": {
            "gpu": best_flops_per_dollar["name"],
            "value": best_flops_per_dollar["flops_per_dollar"],
            "at_price": best_flops_per_dollar["cheapest_price"],
            "provider": best_flops_per_dollar["cheapest_provider"]
        },
        "best_vram_per_dollar": {
            "gpu": best_vram_per_dollar["name"],
            "value": best_vram_per_dollar["vram_per_dollar"],
            "at_price": best_vram_per_dollar["cheapest_price"],
            "provider": best_vram_per_dollar["cheapest_provider"]
        },
        "biggest_price_drop": {
            "gpu": biggest_price_drop["name"],
            "change_pct": biggest_price_drop["monthly_change_pct"]
        },
        "most_competitive_market": {
            "gpu": most_competitive["name"],
            "num_providers": most_competitive["num_providers"],
            "price_spread_pct": most_competitive["price_spread_pct"]
        },
        "market_indicators": MARKET_INDICATORS,
        "comparison_matrix": comparison
    }


def get_regional_summary() -> dict:
    """Get regional market summary."""
    return REGIONAL_DATA


def get_workload_recommendations() -> dict:
    """Get workload-based recommendations with current pricing."""
    enriched = {}
    for workload, rec in WORKLOAD_RECOMMENDATIONS.items():
        gpu_prices = {}
        for gpu_id in rec["recommended"]:
            prices = get_cheapest_by_gpu(gpu_id)
            if prices:
                gpu_prices[gpu_id] = {
                    "cheapest": prices[0]["price_per_gpu_hr"],
                    "provider": prices[0]["provider"],
                    "monthly_1gpu": round(prices[0]["price_per_gpu_hr"] * 730, 2)
                }
        enriched[workload] = {**rec, "current_prices": gpu_prices}
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: GPU Utilization & Efficiency Metrics
# ──────────────────────────────────────────────────────────────────────────────

UTILIZATION_METRICS = {
    "AWS": {
        "H100-SXM": {"avg_utilization_pct": 78, "peak_pct": 94, "off_peak_pct": 52, "idle_cost_per_hr": 0.48, "efficiency_score": 82, "utilization_trend": [68, 71, 74, 76, 78]},
        "B200": {"avg_utilization_pct": 85, "peak_pct": 97, "off_peak_pct": 61, "idle_cost_per_hr": 0.72, "efficiency_score": 88, "utilization_trend": [72, 76, 80, 83, 85]},
        "A100-80GB": {"avg_utilization_pct": 65, "peak_pct": 88, "off_peak_pct": 38, "idle_cost_per_hr": 0.42, "efficiency_score": 70, "utilization_trend": [75, 72, 70, 67, 65]},
        "MI300X": {"avg_utilization_pct": 58, "peak_pct": 82, "off_peak_pct": 30, "idle_cost_per_hr": 0.55, "efficiency_score": 62, "utilization_trend": [40, 45, 50, 54, 58]}
    },
    "GCP": {
        "H100-SXM": {"avg_utilization_pct": 76, "peak_pct": 93, "off_peak_pct": 50, "idle_cost_per_hr": 0.52, "efficiency_score": 80, "utilization_trend": [66, 69, 72, 74, 76]},
        "B200": {"avg_utilization_pct": 83, "peak_pct": 96, "off_peak_pct": 58, "idle_cost_per_hr": 0.78, "efficiency_score": 86, "utilization_trend": [70, 74, 78, 81, 83]},
        "A100-80GB": {"avg_utilization_pct": 63, "peak_pct": 86, "off_peak_pct": 36, "idle_cost_per_hr": 0.45, "efficiency_score": 68, "utilization_trend": [73, 70, 68, 65, 63]},
        "MI300X": {"avg_utilization_pct": 55, "peak_pct": 80, "off_peak_pct": 28, "idle_cost_per_hr": 0.58, "efficiency_score": 59, "utilization_trend": [38, 42, 47, 51, 55]}
    },
    "Azure": {
        "H100-SXM": {"avg_utilization_pct": 74, "peak_pct": 92, "off_peak_pct": 48, "idle_cost_per_hr": 0.55, "efficiency_score": 78, "utilization_trend": [64, 67, 70, 72, 74]},
        "B200": {"avg_utilization_pct": 81, "peak_pct": 95, "off_peak_pct": 55, "idle_cost_per_hr": 0.82, "efficiency_score": 84, "utilization_trend": [68, 72, 76, 79, 81]},
        "A100-80GB": {"avg_utilization_pct": 62, "peak_pct": 85, "off_peak_pct": 35, "idle_cost_per_hr": 0.47, "efficiency_score": 67, "utilization_trend": [72, 69, 67, 64, 62]},
        "MI300X": {"avg_utilization_pct": 60, "peak_pct": 84, "off_peak_pct": 32, "idle_cost_per_hr": 0.52, "efficiency_score": 64, "utilization_trend": [42, 47, 52, 56, 60]}
    },
    "Lambda": {
        "H100-SXM": {"avg_utilization_pct": 82, "peak_pct": 96, "off_peak_pct": 60, "idle_cost_per_hr": 0.38, "efficiency_score": 87, "utilization_trend": [72, 75, 78, 80, 82]},
        "B200": {"avg_utilization_pct": 88, "peak_pct": 98, "off_peak_pct": 68, "idle_cost_per_hr": 0.58, "efficiency_score": 91, "utilization_trend": [76, 80, 84, 86, 88]},
        "A100-80GB": {"avg_utilization_pct": 70, "peak_pct": 90, "off_peak_pct": 42, "idle_cost_per_hr": 0.32, "efficiency_score": 75, "utilization_trend": [78, 76, 74, 72, 70]},
        "MI300X": {"avg_utilization_pct": 62, "peak_pct": 85, "off_peak_pct": 34, "idle_cost_per_hr": 0.48, "efficiency_score": 66, "utilization_trend": [44, 49, 54, 58, 62]}
    },
    "CoreWeave": {
        "H100-SXM": {"avg_utilization_pct": 84, "peak_pct": 97, "off_peak_pct": 62, "idle_cost_per_hr": 0.35, "efficiency_score": 89, "utilization_trend": [74, 77, 80, 82, 84]},
        "B200": {"avg_utilization_pct": 90, "peak_pct": 99, "off_peak_pct": 72, "idle_cost_per_hr": 0.52, "efficiency_score": 93, "utilization_trend": [78, 82, 86, 88, 90]},
        "A100-80GB": {"avg_utilization_pct": 68, "peak_pct": 89, "off_peak_pct": 40, "idle_cost_per_hr": 0.28, "efficiency_score": 73, "utilization_trend": [76, 74, 72, 70, 68]},
        "MI300X": {"avg_utilization_pct": 64, "peak_pct": 86, "off_peak_pct": 36, "idle_cost_per_hr": 0.44, "efficiency_score": 68, "utilization_trend": [46, 51, 56, 60, 64]}
    },
    "RunPod": {
        "H100-SXM": {"avg_utilization_pct": 80, "peak_pct": 95, "off_peak_pct": 56, "idle_cost_per_hr": 0.40, "efficiency_score": 85, "utilization_trend": [70, 73, 76, 78, 80]},
        "B200": {"avg_utilization_pct": 86, "peak_pct": 97, "off_peak_pct": 64, "idle_cost_per_hr": 0.60, "efficiency_score": 89, "utilization_trend": [74, 78, 82, 84, 86]},
        "A100-80GB": {"avg_utilization_pct": 66, "peak_pct": 87, "off_peak_pct": 38, "idle_cost_per_hr": 0.30, "efficiency_score": 71, "utilization_trend": [74, 72, 70, 68, 66]},
        "MI300X": {"avg_utilization_pct": 56, "peak_pct": 81, "off_peak_pct": 29, "idle_cost_per_hr": 0.50, "efficiency_score": 60, "utilization_trend": [38, 43, 48, 52, 56]}
    },
    "Vast.ai": {
        "H100-SXM": {"avg_utilization_pct": 71, "peak_pct": 90, "off_peak_pct": 44, "idle_cost_per_hr": 0.32, "efficiency_score": 76, "utilization_trend": [61, 64, 67, 69, 71]},
        "B200": {"avg_utilization_pct": 78, "peak_pct": 94, "off_peak_pct": 52, "idle_cost_per_hr": 0.50, "efficiency_score": 82, "utilization_trend": [66, 70, 74, 76, 78]},
        "A100-80GB": {"avg_utilization_pct": 64, "peak_pct": 86, "off_peak_pct": 36, "idle_cost_per_hr": 0.22, "efficiency_score": 69, "utilization_trend": [72, 70, 68, 66, 64]},
        "MI300X": {"avg_utilization_pct": 50, "peak_pct": 76, "off_peak_pct": 24, "idle_cost_per_hr": 0.42, "efficiency_score": 54, "utilization_trend": [34, 38, 42, 46, 50]}
    },
    "FluidStack": {
        "H100-SXM": {"avg_utilization_pct": 69, "peak_pct": 88, "off_peak_pct": 42, "idle_cost_per_hr": 0.30, "efficiency_score": 74, "utilization_trend": [59, 62, 65, 67, 69]},
        "B200": {"avg_utilization_pct": 76, "peak_pct": 93, "off_peak_pct": 50, "idle_cost_per_hr": 0.48, "efficiency_score": 80, "utilization_trend": [64, 68, 72, 74, 76]},
        "A100-80GB": {"avg_utilization_pct": 62, "peak_pct": 84, "off_peak_pct": 34, "idle_cost_per_hr": 0.20, "efficiency_score": 67, "utilization_trend": [70, 68, 66, 64, 62]},
        "MI300X": {"avg_utilization_pct": 48, "peak_pct": 74, "off_peak_pct": 22, "idle_cost_per_hr": 0.40, "efficiency_score": 52, "utilization_trend": [32, 36, 40, 44, 48]}
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: Capacity Reservation & Commitment Analytics
# ──────────────────────────────────────────────────────────────────────────────

RESERVATION_ANALYTICS = {
    "H100-SXM": {
        "on_demand_rate": 2.18,
        "spot_avg_rate": 0.98,
        "reserved_1yr_rate": 1.53,
        "reserved_3yr_rate": 0.87,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper when available"},
        "breakeven_hours_1yr": {"monthly_hrs": 438, "description": "60% utilization to break even on 1yr reserved vs on-demand"},
        "breakeven_hours_3yr": {"monthly_hrs": 292, "description": "40% utilization to break even on 3yr reserved vs on-demand"},
        "savings_at_utilization": {
            "40_pct": {"spot": 55, "reserved_1yr": -8, "reserved_3yr": 22},
            "60_pct": {"spot": 55, "reserved_1yr": 12, "reserved_3yr": 42},
            "80_pct": {"spot": 55, "reserved_1yr": 25, "reserved_3yr": 55},
            "100_pct": {"spot": 55, "reserved_1yr": 30, "reserved_3yr": 60}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "Best value under 50% utilization, accept interruption risk"},
            "medium_util": {"type": "reserved_1yr", "reason": "12-25% savings at 60-80% utilization with guaranteed capacity"},
            "high_util": {"type": "reserved_3yr", "reason": "Up to 60% savings at sustained high utilization"}
        }
    },
    "B200": {
        "on_demand_rate": 4.25,
        "spot_avg_rate": 2.12,
        "reserved_1yr_rate": 3.19,
        "reserved_3yr_rate": 1.70,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper when available"},
        "breakeven_hours_1yr": {"monthly_hrs": 450, "description": "62% utilization to break even"},
        "breakeven_hours_3yr": {"monthly_hrs": 300, "description": "41% utilization to break even"},
        "savings_at_utilization": {
            "40_pct": {"spot": 50, "reserved_1yr": -10, "reserved_3yr": 18},
            "60_pct": {"spot": 50, "reserved_1yr": 10, "reserved_3yr": 38},
            "80_pct": {"spot": 50, "reserved_1yr": 22, "reserved_3yr": 52},
            "100_pct": {"spot": 50, "reserved_1yr": 25, "reserved_3yr": 60}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "50% savings, limited availability for B200 spot"},
            "medium_util": {"type": "reserved_1yr", "reason": "Guaranteed Blackwell capacity, moderate savings"},
            "high_util": {"type": "reserved_3yr", "reason": "Lock in next-gen pricing before further demand increases"}
        }
    },
    "A100-80GB": {
        "on_demand_rate": 1.10,
        "spot_avg_rate": 0.40,
        "reserved_1yr_rate": 0.77,
        "reserved_3yr_rate": 0.44,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper"},
        "breakeven_hours_1yr": {"monthly_hrs": 420, "description": "58% utilization to break even"},
        "breakeven_hours_3yr": {"monthly_hrs": 270, "description": "37% utilization to break even"},
        "savings_at_utilization": {
            "40_pct": {"spot": 64, "reserved_1yr": -5, "reserved_3yr": 28},
            "60_pct": {"spot": 64, "reserved_1yr": 15, "reserved_3yr": 45},
            "80_pct": {"spot": 64, "reserved_1yr": 28, "reserved_3yr": 58},
            "100_pct": {"spot": 64, "reserved_1yr": 30, "reserved_3yr": 60}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "Abundant spot availability, 64% savings"},
            "medium_util": {"type": "spot", "reason": "A100 spot is reliable enough for medium workloads"},
            "high_util": {"type": "reserved_1yr", "reason": "Avoid 3yr lock-in on aging hardware"}
        }
    },
    "MI300X": {
        "on_demand_rate": 1.72,
        "spot_avg_rate": 0.69,
        "reserved_1yr_rate": 1.20,
        "reserved_3yr_rate": 0.69,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper"},
        "breakeven_hours_1yr": {"monthly_hrs": 430, "description": "59% utilization to break even"},
        "breakeven_hours_3yr": {"monthly_hrs": 280, "description": "38% utilization to break even"},
        "savings_at_utilization": {
            "40_pct": {"spot": 60, "reserved_1yr": -6, "reserved_3yr": 25},
            "60_pct": {"spot": 60, "reserved_1yr": 13, "reserved_3yr": 42},
            "80_pct": {"spot": 60, "reserved_1yr": 26, "reserved_3yr": 56},
            "100_pct": {"spot": 60, "reserved_1yr": 30, "reserved_3yr": 60}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "Good spot savings, growing AMD availability"},
            "medium_util": {"type": "reserved_1yr", "reason": "Lock in AMD pricing advantage vs NVIDIA"},
            "high_util": {"type": "reserved_3yr", "reason": "Best TCO for AMD-compatible workloads"}
        }
    },
    "H200": {
        "on_demand_rate": 3.50,
        "spot_avg_rate": 1.58,
        "reserved_1yr_rate": 2.45,
        "reserved_3yr_rate": 1.40,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper"},
        "breakeven_hours_1yr": {"monthly_hrs": 445, "description": "61% utilization to break even"},
        "breakeven_hours_3yr": {"monthly_hrs": 290, "description": "40% utilization to break even"},
        "savings_at_utilization": {
            "40_pct": {"spot": 55, "reserved_1yr": -9, "reserved_3yr": 20},
            "60_pct": {"spot": 55, "reserved_1yr": 11, "reserved_3yr": 40},
            "80_pct": {"spot": 55, "reserved_1yr": 24, "reserved_3yr": 54},
            "100_pct": {"spot": 55, "reserved_1yr": 30, "reserved_3yr": 60}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "H200 spot increasingly available"},
            "medium_util": {"type": "reserved_1yr", "reason": "Good bridge GPU before Blackwell ramp"},
            "high_util": {"type": "reserved_1yr", "reason": "Avoid 3yr on transitional hardware"}
        }
    },
    "RTX-4090": {
        "on_demand_rate": 0.22,
        "spot_avg_rate": 0.11,
        "reserved_1yr_rate": 0.15,
        "reserved_3yr_rate": 0.09,
        "breakeven_hours_spot": {"vs_on_demand": 0, "description": "Spot always cheaper"},
        "breakeven_hours_1yr": {"monthly_hrs": 400, "description": "55% utilization to break even"},
        "breakeven_hours_3yr": {"monthly_hrs": 250, "description": "34% utilization to break even"},
        "savings_at_utilization": {
            "40_pct": {"spot": 50, "reserved_1yr": -3, "reserved_3yr": 32},
            "60_pct": {"spot": 50, "reserved_1yr": 18, "reserved_3yr": 48},
            "80_pct": {"spot": 50, "reserved_1yr": 30, "reserved_3yr": 58},
            "100_pct": {"spot": 50, "reserved_1yr": 32, "reserved_3yr": 59}
        },
        "recommended_commitment": {
            "low_util": {"type": "spot", "reason": "Consumer GPU spot is very cheap"},
            "medium_util": {"type": "spot", "reason": "Spot reliability high for consumer GPUs"},
            "high_util": {"type": "reserved_1yr", "reason": "Consumer GPUs may phase out, avoid 3yr"}
        }
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: Price Elasticity & Forecasting
# ──────────────────────────────────────────────────────────────────────────────

PRICE_FORECASTS = {
    "H100-SXM": {
        "current_avg": 2.18,
        "elasticity_coefficient": -0.35,
        "forecast_3mo": {"low": 1.85, "mid": 1.95, "high": 2.10, "confidence": 0.78},
        "forecast_6mo": {"low": 1.55, "mid": 1.72, "high": 1.95, "confidence": 0.65},
        "forecast_12mo": {"low": 1.20, "mid": 1.48, "high": 1.80, "confidence": 0.48},
        "price_floor": 1.10,
        "supply_signal": "increasing",
        "demand_signal": "stable",
        "pattern_match": "B-curve decline (Blackwell displacement)"
    },
    "B200": {
        "current_avg": 4.25,
        "elasticity_coefficient": -0.18,
        "forecast_3mo": {"low": 3.80, "mid": 4.05, "high": 4.30, "confidence": 0.72},
        "forecast_6mo": {"low": 3.20, "mid": 3.60, "high": 4.10, "confidence": 0.58},
        "forecast_12mo": {"low": 2.50, "mid": 3.10, "high": 3.80, "confidence": 0.42},
        "price_floor": 2.20,
        "supply_signal": "constrained",
        "demand_signal": "strong",
        "pattern_match": "Early-cycle premium (mirrors H100 2023 trajectory)"
    },
    "A100-80GB": {
        "current_avg": 1.10,
        "elasticity_coefficient": -0.52,
        "forecast_3mo": {"low": 0.88, "mid": 0.98, "high": 1.08, "confidence": 0.82},
        "forecast_6mo": {"low": 0.70, "mid": 0.85, "high": 1.00, "confidence": 0.70},
        "forecast_12mo": {"low": 0.55, "mid": 0.72, "high": 0.90, "confidence": 0.55},
        "price_floor": 0.45,
        "supply_signal": "surplus",
        "demand_signal": "declining",
        "pattern_match": "Late-cycle depreciation (2-gen behind)"
    },
    "H200": {
        "current_avg": 3.50,
        "elasticity_coefficient": -0.28,
        "forecast_3mo": {"low": 3.05, "mid": 3.25, "high": 3.50, "confidence": 0.75},
        "forecast_6mo": {"low": 2.60, "mid": 2.90, "high": 3.30, "confidence": 0.62},
        "forecast_12mo": {"low": 2.00, "mid": 2.45, "high": 3.00, "confidence": 0.45},
        "price_floor": 1.80,
        "supply_signal": "increasing",
        "demand_signal": "moderate",
        "pattern_match": "Mid-cycle transition (squeezed by B200 above, H100 below)"
    },
    "MI300X": {
        "current_avg": 1.72,
        "elasticity_coefficient": -0.42,
        "forecast_3mo": {"low": 1.45, "mid": 1.58, "high": 1.72, "confidence": 0.74},
        "forecast_6mo": {"low": 1.20, "mid": 1.40, "high": 1.65, "confidence": 0.60},
        "forecast_12mo": {"low": 0.95, "mid": 1.20, "high": 1.50, "confidence": 0.46},
        "price_floor": 0.80,
        "supply_signal": "increasing",
        "demand_signal": "growing",
        "pattern_match": "Competitive pressure play (AMD gaining share)"
    },
    "RTX-4090": {
        "current_avg": 0.22,
        "elasticity_coefficient": -0.48,
        "forecast_3mo": {"low": 0.18, "mid": 0.20, "high": 0.23, "confidence": 0.76},
        "forecast_6mo": {"low": 0.14, "mid": 0.17, "high": 0.22, "confidence": 0.62},
        "forecast_12mo": {"low": 0.10, "mid": 0.14, "high": 0.20, "confidence": 0.48},
        "price_floor": 0.08,
        "supply_signal": "stable",
        "demand_signal": "niche",
        "pattern_match": "Consumer surplus (RTX 5090 launch pressure)"
    },
    "GB200": {
        "current_avg": 7.50,
        "elasticity_coefficient": -0.10,
        "forecast_3mo": {"low": 6.80, "mid": 7.20, "high": 7.60, "confidence": 0.65},
        "forecast_6mo": {"low": 5.80, "mid": 6.50, "high": 7.30, "confidence": 0.50},
        "forecast_12mo": {"low": 4.50, "mid": 5.50, "high": 6.80, "confidence": 0.35},
        "price_floor": 3.80,
        "supply_signal": "very_constrained",
        "demand_signal": "very_strong",
        "pattern_match": "Launch premium (NVL72 rack-scale, limited supply)"
    },
    "MI325X": {
        "current_avg": 2.10,
        "elasticity_coefficient": -0.38,
        "forecast_3mo": {"low": 1.80, "mid": 1.95, "high": 2.12, "confidence": 0.70},
        "forecast_6mo": {"low": 1.50, "mid": 1.72, "high": 2.00, "confidence": 0.56},
        "forecast_12mo": {"low": 1.15, "mid": 1.45, "high": 1.85, "confidence": 0.42},
        "price_floor": 0.95,
        "supply_signal": "increasing",
        "demand_signal": "growing",
        "pattern_match": "AMD refresh cycle (MI300X successor, competitive pressure)"
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature 4: Competitive Moat Tracker
# ──────────────────────────────────────────────────────────────────────────────

COMPETITIVE_MOAT = {
    "NVIDIA": {
        "performance_score": 95,
        "ecosystem_maturity": 98,
        "software_compatibility": 99,
        "price_performance_ratio": 72,
        "moat_strength_score": 92,
        "market_share_pct": 78,
        "market_share_trend": [88, 86, 84, 81, 78],
        "key_products": ["B200", "GB200", "H200", "H100-SXM"],
        "strengths": ["CUDA ecosystem lock-in", "NVLink/NVSwitch interconnect", "Dominant software stack", "Training performance leadership"],
        "weaknesses": ["Premium pricing", "Supply constraints on latest gen", "Growing competitive pressure"],
        "parity_timeline": None
    },
    "AMD": {
        "performance_score": 78,
        "ecosystem_maturity": 62,
        "software_compatibility": 58,
        "price_performance_ratio": 88,
        "moat_strength_score": 48,
        "market_share_pct": 22,
        "market_share_trend": [12, 14, 16, 19, 22],
        "key_products": ["MI300X", "MI325X", "MI350X"],
        "strengths": ["Price/perf advantage", "Large HBM capacity", "Open ROCm ecosystem", "Rapid market share growth"],
        "weaknesses": ["ROCm maturity gap", "Limited training adoption", "Smaller ecosystem"],
        "parity_timeline": "2027-Q2 for inference, 2028+ for training"
    },
    "Google_TPU": {
        "performance_score": 82,
        "ecosystem_maturity": 75,
        "software_compatibility": 45,
        "price_performance_ratio": 85,
        "moat_strength_score": 55,
        "market_share_pct": 8,
        "market_share_trend": [5, 5, 6, 7, 8],
        "key_products": ["TPU v5p", "TPU v5e", "TPU v6e (Trillium)"],
        "strengths": ["Vertically integrated (GCP only)", "Excellent JAX/TF performance", "Competitive pricing", "Large-scale training proven"],
        "weaknesses": ["GCP lock-in", "No PyTorch native support", "Limited availability outside Google"],
        "parity_timeline": "Niche — competes in JAX/TF workloads only"
    },
    "AWS_Trainium": {
        "performance_score": 68,
        "ecosystem_maturity": 42,
        "software_compatibility": 35,
        "price_performance_ratio": 90,
        "moat_strength_score": 35,
        "market_share_pct": 4,
        "market_share_trend": [1, 2, 2, 3, 4],
        "key_products": ["Trainium2", "Inferentia2"],
        "strengths": ["Aggressive pricing", "AWS ecosystem integration", "Neuron SDK improving", "Cost leadership strategy"],
        "weaknesses": ["Limited model compatibility", "Early ecosystem", "Performance gaps on complex models"],
        "parity_timeline": "2028+ for broad adoption"
    },
    "Intel": {
        "performance_score": 45,
        "ecosystem_maturity": 38,
        "software_compatibility": 40,
        "price_performance_ratio": 65,
        "moat_strength_score": 22,
        "market_share_pct": 2,
        "market_share_trend": [4, 4, 3, 3, 2],
        "key_products": ["Gaudi 3", "Gaudi 2"],
        "strengths": ["Competitive Gaudi 3 pricing", "x86 ecosystem familiarity", "Enterprise relationships"],
        "weaknesses": ["Poor market traction", "Software maturity issues", "Shrinking share", "Strategic uncertainty"],
        "parity_timeline": "Unlikely to achieve broad parity"
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature 5: Energy & Sustainability Index
# ──────────────────────────────────────────────────────────────────────────────

SUSTAINABILITY_INDEX = {
    "AWS": {
        "us-east-1": {"pue": 1.10, "carbon_gco2_per_kwh": 380, "green_energy_pct": 65, "water_usage_l_per_kwh": 1.8, "sustainability_score": 72},
        "us-west-2": {"pue": 1.08, "carbon_gco2_per_kwh": 120, "green_energy_pct": 90, "water_usage_l_per_kwh": 0.9, "sustainability_score": 92},
        "eu-west-1": {"pue": 1.12, "carbon_gco2_per_kwh": 280, "green_energy_pct": 78, "water_usage_l_per_kwh": 1.4, "sustainability_score": 80},
        "eu-north-1": {"pue": 1.06, "carbon_gco2_per_kwh": 45, "green_energy_pct": 98, "water_usage_l_per_kwh": 0.3, "sustainability_score": 97},
        "ap-northeast-1": {"pue": 1.15, "carbon_gco2_per_kwh": 480, "green_energy_pct": 35, "water_usage_l_per_kwh": 2.2, "sustainability_score": 55}
    },
    "GCP": {
        "us-central1": {"pue": 1.08, "carbon_gco2_per_kwh": 350, "green_energy_pct": 72, "water_usage_l_per_kwh": 1.5, "sustainability_score": 78},
        "us-west1": {"pue": 1.06, "carbon_gco2_per_kwh": 100, "green_energy_pct": 95, "water_usage_l_per_kwh": 0.7, "sustainability_score": 95},
        "europe-west4": {"pue": 1.09, "carbon_gco2_per_kwh": 320, "green_energy_pct": 80, "water_usage_l_per_kwh": 1.3, "sustainability_score": 82},
        "europe-north1": {"pue": 1.05, "carbon_gco2_per_kwh": 30, "green_energy_pct": 99, "water_usage_l_per_kwh": 0.2, "sustainability_score": 98},
        "asia-east1": {"pue": 1.14, "carbon_gco2_per_kwh": 550, "green_energy_pct": 28, "water_usage_l_per_kwh": 2.5, "sustainability_score": 48}
    },
    "Azure": {
        "eastus": {"pue": 1.12, "carbon_gco2_per_kwh": 390, "green_energy_pct": 60, "water_usage_l_per_kwh": 1.9, "sustainability_score": 68},
        "westus2": {"pue": 1.09, "carbon_gco2_per_kwh": 140, "green_energy_pct": 88, "water_usage_l_per_kwh": 1.0, "sustainability_score": 90},
        "northeurope": {"pue": 1.07, "carbon_gco2_per_kwh": 60, "green_energy_pct": 96, "water_usage_l_per_kwh": 0.4, "sustainability_score": 96},
        "westeurope": {"pue": 1.10, "carbon_gco2_per_kwh": 300, "green_energy_pct": 75, "water_usage_l_per_kwh": 1.5, "sustainability_score": 78},
        "japaneast": {"pue": 1.16, "carbon_gco2_per_kwh": 470, "green_energy_pct": 38, "water_usage_l_per_kwh": 2.1, "sustainability_score": 56}
    },
    "CoreWeave": {
        "us-east": {"pue": 1.15, "carbon_gco2_per_kwh": 400, "green_energy_pct": 55, "water_usage_l_per_kwh": 2.0, "sustainability_score": 65},
        "us-west": {"pue": 1.10, "carbon_gco2_per_kwh": 150, "green_energy_pct": 85, "water_usage_l_per_kwh": 1.1, "sustainability_score": 88},
        "eu-west": {"pue": 1.12, "carbon_gco2_per_kwh": 290, "green_energy_pct": 76, "water_usage_l_per_kwh": 1.5, "sustainability_score": 79}
    },
    "Lambda": {
        "us-south": {"pue": 1.18, "carbon_gco2_per_kwh": 420, "green_energy_pct": 50, "water_usage_l_per_kwh": 2.2, "sustainability_score": 60},
        "us-west": {"pue": 1.11, "carbon_gco2_per_kwh": 160, "green_energy_pct": 82, "water_usage_l_per_kwh": 1.2, "sustainability_score": 85}
    }
}

GPU_CARBON_FOOTPRINT = {
    "H100-SXM": {"tdp_watts": 700, "typical_watts": 580, "kwh_per_hour": 0.58, "annual_kwh_full_util": 5081, "carbon_kg_per_year_us_avg": 2032, "carbon_kg_per_year_eu_nordic": 228, "water_liters_per_year_us_avg": 9146, "embodied_carbon_kg": 150},
    "B200": {"tdp_watts": 1000, "typical_watts": 820, "kwh_per_hour": 0.82, "annual_kwh_full_util": 7183, "carbon_kg_per_year_us_avg": 2873, "carbon_kg_per_year_eu_nordic": 323, "water_liters_per_year_us_avg": 12930, "embodied_carbon_kg": 200},
    "A100-80GB": {"tdp_watts": 300, "typical_watts": 250, "kwh_per_hour": 0.25, "annual_kwh_full_util": 2190, "carbon_kg_per_year_us_avg": 876, "carbon_kg_per_year_eu_nordic": 99, "water_liters_per_year_us_avg": 3942, "embodied_carbon_kg": 100},
    "MI300X": {"tdp_watts": 750, "typical_watts": 620, "kwh_per_hour": 0.62, "annual_kwh_full_util": 5431, "carbon_kg_per_year_us_avg": 2172, "carbon_kg_per_year_eu_nordic": 244, "water_liters_per_year_us_avg": 9776, "embodied_carbon_kg": 160},
    "H200": {"tdp_watts": 700, "typical_watts": 580, "kwh_per_hour": 0.58, "annual_kwh_full_util": 5081, "carbon_kg_per_year_us_avg": 2032, "carbon_kg_per_year_eu_nordic": 228, "water_liters_per_year_us_avg": 9146, "embodied_carbon_kg": 155},
    "RTX-4090": {"tdp_watts": 450, "typical_watts": 370, "kwh_per_hour": 0.37, "annual_kwh_full_util": 3241, "carbon_kg_per_year_us_avg": 1296, "carbon_kg_per_year_eu_nordic": 146, "water_liters_per_year_us_avg": 5834, "embodied_carbon_kg": 80}
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature 6: Supply Chain Risk Dashboard
# ──────────────────────────────────────────────────────────────────────────────

SUPPLY_CHAIN_RISK = {
    "NVIDIA": {
        "supply_risk_score": 35,
        "tsmc_dependency_pct": 100,
        "geopolitical_risk": "medium",
        "lead_time_weeks": 1,
        "lead_time_trend": [52, 36, 20, 8, 1],
        "export_control_impact": "moderate",
        "bottlenecks": ["TSMC CoWoS packaging", "HBM3e supply (SK Hynix/Samsung)", "NVLink switch availability"],
        "risk_trend": "improving",
        "single_source_components": ["CoWoS packaging (TSMC)", "NVSwitch (TSMC 4nm)"],
        "inventory_weeks": 6
    },
    "AMD": {
        "supply_risk_score": 42,
        "tsmc_dependency_pct": 100,
        "geopolitical_risk": "medium",
        "lead_time_weeks": 3,
        "lead_time_trend": [24, 18, 12, 6, 3],
        "export_control_impact": "moderate",
        "bottlenecks": ["TSMC 5nm/3nm capacity", "HBM3 supply allocation", "ROCm software readiness"],
        "risk_trend": "improving",
        "single_source_components": ["Advanced packaging (TSMC)", "HBM3 (SK Hynix/Micron)"],
        "inventory_weeks": 8
    },
    "Google_TPU": {
        "supply_risk_score": 28,
        "tsmc_dependency_pct": 100,
        "geopolitical_risk": "low",
        "lead_time_weeks": 0,
        "lead_time_trend": [4, 3, 2, 1, 0],
        "export_control_impact": "low",
        "bottlenecks": ["TSMC wafer allocation", "Internal demand vs cloud availability"],
        "risk_trend": "stable",
        "single_source_components": ["TPU dies (Broadcom design, TSMC fab)"],
        "inventory_weeks": 12
    },
    "Intel": {
        "supply_risk_score": 55,
        "tsmc_dependency_pct": 30,
        "geopolitical_risk": "low",
        "lead_time_weeks": 2,
        "lead_time_trend": [8, 6, 4, 3, 2],
        "export_control_impact": "low",
        "bottlenecks": ["Intel Foundry yield issues", "Gaudi software ecosystem", "Customer confidence"],
        "risk_trend": "worsening",
        "single_source_components": ["Gaudi dies (TSMC 5nm)"],
        "inventory_weeks": 14
    }
}

EXPORT_CONTROL_TRACKER = [
    {"date": "2022-10", "regulation": "US CHIPS Act Export Controls", "target": "China", "impact": "high", "affected_gpus": ["A100", "H100"], "description": "Initial restrictions on advanced AI chips to China"},
    {"date": "2023-10", "regulation": "Updated Export Controls", "target": "China + 40 countries", "impact": "high", "affected_gpus": ["A100", "H100", "MI300X", "L40S"], "description": "Closed loopholes, expanded country list, compute density thresholds"},
    {"date": "2024-01", "regulation": "NVIDIA China-specific SKUs", "target": "China", "impact": "medium", "affected_gpus": ["H20", "L20"], "description": "Compliance variants with reduced specs for China market"},
    {"date": "2024-09", "regulation": "EU AI Act Phase 1", "target": "EU", "impact": "low", "affected_gpus": [], "description": "Risk-based AI regulation, compute reporting requirements for GPAI"},
    {"date": "2025-03", "regulation": "Diffusion Rule (Biden framework)", "target": "Global tiers", "impact": "high", "affected_gpus": ["B200", "GB200", "MI325X"], "description": "3-tier country framework for AI chip exports, datacenter caps"},
    {"date": "2025-07", "regulation": "Japan/Netherlands ASML restrictions", "target": "China", "impact": "medium", "affected_gpus": [], "description": "DUV lithography equipment restrictions aligned with US policy"},
    {"date": "2026-01", "regulation": "Proposed Compute Sovereignty Act", "target": "US domestic", "impact": "medium", "affected_gpus": ["B200", "GB200"], "description": "Proposed requirements for domestic AI compute capacity reservations"}
]


# ──────────────────────────────────────────────────────────────────────────────
# Feature 7: Model-to-Hardware Fit Matrix
# ──────────────────────────────────────────────────────────────────────────────

MODEL_HARDWARE_FIT = {
    "7B": {
        "models": ["Llama-3.1-8B", "Mistral-7B", "Qwen2.5-7B"],
        "vram_required_gb": 14,
        "gpus": {
            "H100-SXM": {"optimal_config": "1x H100", "batch_size": 256, "throughput_tok_s": 450, "cost_per_1m_tokens": 0.055, "vram_headroom_pct": 82, "fit_score": 65, "notes": "Overkill for 7B — wastes VRAM"},
            "B200": {"optimal_config": "1x B200", "batch_size": 512, "throughput_tok_s": 680, "cost_per_1m_tokens": 0.042, "vram_headroom_pct": 93, "fit_score": 55, "notes": "Massive overkill, only if bundled"},
            "A100-80GB": {"optimal_config": "1x A100", "batch_size": 128, "throughput_tok_s": 280, "cost_per_1m_tokens": 0.048, "vram_headroom_pct": 82, "fit_score": 72, "notes": "Good balance for small models"},
            "MI300X": {"optimal_config": "1x MI300X", "batch_size": 256, "throughput_tok_s": 380, "cost_per_1m_tokens": 0.045, "vram_headroom_pct": 93, "fit_score": 60, "notes": "VRAM overkill, decent throughput"},
            "RTX-4090": {"optimal_config": "1x RTX-4090", "batch_size": 64, "throughput_tok_s": 140, "cost_per_1m_tokens": 0.035, "vram_headroom_pct": 42, "fit_score": 92, "notes": "Best cost/perf for 7B inference"}
        }
    },
    "13B": {
        "models": ["Llama-3.1-13B", "CodeLlama-13B", "Qwen2.5-14B"],
        "vram_required_gb": 26,
        "gpus": {
            "H100-SXM": {"optimal_config": "1x H100", "batch_size": 128, "throughput_tok_s": 320, "cost_per_1m_tokens": 0.078, "vram_headroom_pct": 68, "fit_score": 75, "notes": "Good balance of speed and cost"},
            "B200": {"optimal_config": "1x B200", "batch_size": 256, "throughput_tok_s": 480, "cost_per_1m_tokens": 0.062, "vram_headroom_pct": 86, "fit_score": 65, "notes": "Overkill but fast"},
            "A100-80GB": {"optimal_config": "1x A100", "batch_size": 64, "throughput_tok_s": 195, "cost_per_1m_tokens": 0.072, "vram_headroom_pct": 68, "fit_score": 82, "notes": "Sweet spot for 13B inference"},
            "MI300X": {"optimal_config": "1x MI300X", "batch_size": 128, "throughput_tok_s": 270, "cost_per_1m_tokens": 0.065, "vram_headroom_pct": 86, "fit_score": 70, "notes": "Good perf, VRAM headroom for batching"},
            "RTX-4090": {"optimal_config": "2x RTX-4090", "batch_size": 32, "throughput_tok_s": 95, "cost_per_1m_tokens": 0.058, "vram_headroom_pct": 8, "fit_score": 72, "notes": "Best cost/perf if 2-GPU setup acceptable"}
        }
    },
    "70B": {
        "models": ["Llama-3.1-70B", "Qwen2.5-72B", "Mixtral-8x22B"],
        "vram_required_gb": 140,
        "gpus": {
            "H100-SXM": {"optimal_config": "2x H100 (NVLink)", "batch_size": 64, "throughput_tok_s": 95, "cost_per_1m_tokens": 0.38, "vram_headroom_pct": 14, "fit_score": 85, "notes": "Standard config for 70B, good perf"},
            "B200": {"optimal_config": "1x B200", "batch_size": 128, "throughput_tok_s": 145, "cost_per_1m_tokens": 0.28, "vram_headroom_pct": 27, "fit_score": 92, "notes": "Single GPU! 192GB VRAM fits 70B"},
            "A100-80GB": {"optimal_config": "2x A100 (NVLink)", "batch_size": 32, "throughput_tok_s": 52, "cost_per_1m_tokens": 0.52, "vram_headroom_pct": 14, "fit_score": 72, "notes": "Viable but slower, tight VRAM"},
            "MI300X": {"optimal_config": "1x MI300X", "batch_size": 64, "throughput_tok_s": 82, "cost_per_1m_tokens": 0.32, "vram_headroom_pct": 27, "fit_score": 88, "notes": "Single GPU fits 70B, best AMD value"},
            "H200": {"optimal_config": "2x H200", "batch_size": 64, "throughput_tok_s": 110, "cost_per_1m_tokens": 0.42, "vram_headroom_pct": 20, "fit_score": 80, "notes": "NVLink pair, good throughput"}
        }
    },
    "405B": {
        "models": ["Llama-3.1-405B", "DBRX-132B", "Falcon-180B"],
        "vram_required_gb": 810,
        "gpus": {
            "H100-SXM": {"optimal_config": "8x H100 (DGX)", "batch_size": 16, "throughput_tok_s": 28, "cost_per_1m_tokens": 2.80, "vram_headroom_pct": 0, "fit_score": 70, "notes": "Full DGX node, tight VRAM, needs offloading"},
            "B200": {"optimal_config": "8x B200 (NVL)", "batch_size": 64, "throughput_tok_s": 65, "cost_per_1m_tokens": 1.85, "vram_headroom_pct": 47, "fit_score": 90, "notes": "1.5TB VRAM, excellent fit for mega-models"},
            "MI300X": {"optimal_config": "8x MI300X", "batch_size": 32, "throughput_tok_s": 35, "cost_per_1m_tokens": 2.20, "vram_headroom_pct": 47, "fit_score": 78, "notes": "1.5TB HBM, competitive AMD option"},
            "GB200": {"optimal_config": "4x GB200", "batch_size": 128, "throughput_tok_s": 95, "cost_per_1m_tokens": 1.40, "vram_headroom_pct": 60, "fit_score": 95, "notes": "NVL72 rack-scale, best for 400B+ models"},
            "H200": {"optimal_config": "8x H200", "batch_size": 16, "throughput_tok_s": 32, "cost_per_1m_tokens": 2.50, "vram_headroom_pct": 10, "fit_score": 65, "notes": "Feasible but H100 successor, limited headroom"}
        }
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions for new features
# ──────────────────────────────────────────────────────────────────────────────

def get_utilization_summary() -> dict:
    """Aggregate utilization metrics across providers and GPUs."""
    gpu_summary = {}
    for provider, gpus in UTILIZATION_METRICS.items():
        for gpu_id, metrics in gpus.items():
            if gpu_id not in gpu_summary:
                gpu_summary[gpu_id] = {"providers": {}, "avg_utilization": 0, "avg_efficiency": 0, "count": 0}
            gpu_summary[gpu_id]["providers"][provider] = metrics
            gpu_summary[gpu_id]["avg_utilization"] += metrics["avg_utilization_pct"]
            gpu_summary[gpu_id]["avg_efficiency"] += metrics["efficiency_score"]
            gpu_summary[gpu_id]["count"] += 1
    for gpu_id in gpu_summary:
        n = gpu_summary[gpu_id]["count"]
        gpu_summary[gpu_id]["avg_utilization"] = round(gpu_summary[gpu_id]["avg_utilization"] / n, 1)
        gpu_summary[gpu_id]["avg_efficiency"] = round(gpu_summary[gpu_id]["avg_efficiency"] / n, 1)
    return gpu_summary


def get_reservation_analysis() -> dict:
    """Get reservation analytics with current pricing context."""
    return RESERVATION_ANALYTICS


def get_price_forecasts() -> dict:
    """Get price forecasts for all tracked GPUs."""
    return PRICE_FORECASTS


def get_competitive_landscape() -> dict:
    """Get competitive moat analysis for all vendors."""
    return COMPETITIVE_MOAT


def get_sustainability_summary() -> dict:
    """Aggregate sustainability data across providers and regions."""
    provider_summary = {}
    for provider, regions in SUSTAINABILITY_INDEX.items():
        scores = [r["sustainability_score"] for r in regions.values()]
        green = [r["green_energy_pct"] for r in regions.values()]
        pues = [r["pue"] for r in regions.values()]
        provider_summary[provider] = {
            "regions": regions,
            "avg_sustainability_score": round(sum(scores) / len(scores), 1),
            "avg_green_energy_pct": round(sum(green) / len(green), 1),
            "avg_pue": round(sum(pues) / len(pues), 2),
            "best_region": max(regions.items(), key=lambda x: x[1]["sustainability_score"])[0],
            "worst_region": min(regions.items(), key=lambda x: x[1]["sustainability_score"])[0]
        }
    return {"providers": provider_summary, "gpu_carbon": GPU_CARBON_FOOTPRINT}


def get_supply_chain_summary() -> dict:
    """Get supply chain risk summary with export control context."""
    return {"vendors": SUPPLY_CHAIN_RISK, "export_controls": EXPORT_CONTROL_TRACKER}


def get_model_hardware_fit() -> dict:
    """Get model-to-hardware fit matrix."""
    return MODEL_HARDWARE_FIT
