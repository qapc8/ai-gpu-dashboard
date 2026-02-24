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
    # --- NVIDIA Ada Lovelace ---
    "L40S": {
        "name": "NVIDIA L40S 48GB",
        "vendor": "NVIDIA",
        "vram_gb": 48,
        "arch": "Ada Lovelace",
        "fp16_tflops": 362,
        "fp32_tflops": 91.6,
        "tdp_watts": 350,
        "interconnect": "PCIe 4.0",
        "release_year": 2023,
        "msrp_usd": 8000,
        "tier": "mid"
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
        "type": "cloud",
        "gpus": {
            "B200": {"instance": "p6-b200.48xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 14.24, "regions": {
                "us-east-1": 14.24, "us-east-2": 14.24, "us-west-2": 14.24,
                "eu-west-1": 15.66, "eu-central-1": 16.24,
                "ap-northeast-1": 17.09, "ap-southeast-1": 16.66
            }},
            "H200": {"instance": "p5en.48xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 7.91, "regions": {
                "us-east-1": 7.91, "us-east-2": 7.91, "us-west-2": 7.91,
                "eu-west-1": 8.70, "eu-central-1": 9.02,
                "ap-northeast-1": 9.50
            }},
            "H100-SXM": {"instance": "p5.48xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 6.88, "regions": {
                "us-east-1": 6.88, "us-east-2": 6.88, "us-west-2": 6.88,
                "eu-west-1": 7.57, "eu-central-1": 7.85,
                "ap-northeast-1": 8.26, "ap-southeast-1": 8.04
            }},
            "A100-80GB": {"instance": "p4de.24xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 3.43, "regions": {
                "us-east-1": 3.43, "us-east-2": 3.43, "us-west-2": 3.43,
                "eu-west-1": 3.77, "eu-central-1": 3.91,
                "ap-northeast-1": 4.12, "ap-southeast-1": 4.01
            }},
            "A100-40GB": {"instance": "p4d.24xlarge", "gpus_per_instance": 8, "price_per_gpu_hr": 2.74, "regions": {
                "us-east-1": 2.74, "us-east-2": 2.74, "us-west-2": 2.74,
                "eu-west-1": 3.01, "eu-central-1": 3.12,
                "ap-northeast-1": 3.29, "ap-southeast-1": 3.20
            }},
            "L40S": {"instance": "g6e.xlarge", "gpus_per_instance": 1, "price_per_gpu_hr": 1.86, "regions": {
                "us-east-1": 1.86, "us-east-2": 1.86, "us-west-2": 1.86,
                "eu-west-1": 2.05, "eu-central-1": 2.12
            }},
            "A10G": {"instance": "g5.xlarge", "gpus_per_instance": 1, "price_per_gpu_hr": 1.006, "regions": {
                "us-east-1": 1.006, "us-east-2": 1.006, "us-west-2": 1.006,
                "eu-west-1": 1.11, "eu-central-1": 1.15,
                "ap-northeast-1": 1.22, "ap-southeast-1": 1.18
            }}
        },
        "reserved_1yr_discount": 0.40,
        "reserved_3yr_discount": 0.60
    },
    "GCP": {
        "provider_name": "Google Cloud Platform",
        "type": "cloud",
        "gpus": {
            "H200": {"instance": "a3-ultragpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 10.85, "regions": {
                "us-central1": 10.85, "us-east4": 10.85, "us-west1": 10.85,
                "europe-west4": 11.94, "europe-west1": 11.82
            }},
            "H100-SXM": {"instance": "a3-highgpu-8g", "gpus_per_instance": 8, "price_per_gpu_hr": 11.06, "regions": {
                "us-central1": 11.06, "us-east4": 11.06, "us-west1": 11.06,
                "europe-west4": 12.17, "europe-west1": 12.04,
                "asia-east1": 12.72, "asia-northeast1": 12.93
            }},
            "A100-80GB": {"instance": "a2-ultragpu-1g", "gpus_per_instance": 1, "price_per_gpu_hr": 5.07, "regions": {
                "us-central1": 5.07, "us-east4": 5.07, "us-west1": 5.07,
                "europe-west4": 5.58, "europe-west1": 5.52,
                "asia-east1": 5.83, "asia-northeast1": 5.93
            }},
            "A100-40GB": {"instance": "a2-highgpu-1g", "gpus_per_instance": 1, "price_per_gpu_hr": 3.67, "regions": {
                "us-central1": 3.67, "us-east4": 3.67, "us-west1": 3.67,
                "europe-west4": 4.04, "europe-west1": 4.00,
                "asia-east1": 4.22, "asia-northeast1": 4.29
            }},
            "V100": {"instance": "n1-standard-8+V100", "gpus_per_instance": 1, "price_per_gpu_hr": 2.97, "regions": {
                "us-central1": 2.97, "us-east4": 2.97, "us-west1": 2.97,
                "europe-west4": 3.27
            }}
        },
        "reserved_1yr_discount": 0.37,
        "reserved_3yr_discount": 0.55
    },
    "Azure": {
        "provider_name": "Microsoft Azure",
        "type": "cloud",
        "gpus": {
            "GB200": {"instance": "ND128isr_NDR_GB200_v6", "gpus_per_instance": 4, "price_per_gpu_hr": 27.04, "regions": {
                "eastus": 27.04, "eastus2": 27.04, "westus2": 27.04,
                "westeurope": 29.74, "northeurope": 29.42
            }},
            "H200": {"instance": "ND96isr_H200_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 10.60, "regions": {
                "eastus": 10.60, "eastus2": 10.60, "westus2": 10.60,
                "westeurope": 11.66, "northeurope": 11.53
            }},
            "H100-SXM": {"instance": "ND96isr_H100_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 12.29, "regions": {
                "eastus": 12.29, "eastus2": 12.29, "westus2": 12.29, "westus3": 12.29,
                "westeurope": 13.52, "northeurope": 13.38,
                "japaneast": 14.75, "southeastasia": 14.34
            }},
            "MI325X": {"instance": "ND96isr_MI325X_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 7.20, "regions": {
                "eastus": 7.20, "eastus2": 7.20, "westus2": 7.20,
                "westeurope": 7.92, "southeastasia": 8.40
            }},
            "MI300X": {"instance": "ND96isr_MI300X_v5", "gpus_per_instance": 8, "price_per_gpu_hr": 6.00, "regions": {
                "eastus": 6.00, "eastus2": 6.00, "westus2": 6.00,
                "westeurope": 6.60, "southeastasia": 7.00
            }},
            "A100-80GB": {"instance": "ND96amsr_A100_v4", "gpus_per_instance": 8, "price_per_gpu_hr": 4.10, "regions": {
                "eastus": 4.10, "eastus2": 4.10, "westus2": 4.10,
                "westeurope": 4.51, "northeurope": 4.46,
                "japaneast": 4.92, "southeastasia": 4.78
            }},
            "A10G": {"instance": "NV36ads_A10_v5", "gpus_per_instance": 1, "price_per_gpu_hr": 0.91, "regions": {
                "eastus": 0.91, "eastus2": 0.91, "westus2": 0.91,
                "westeurope": 1.00, "northeurope": 0.99
            }},
            "V100": {"instance": "NC6s_v3", "gpus_per_instance": 1, "price_per_gpu_hr": 3.06, "regions": {
                "eastus": 3.06, "eastus2": 3.06, "westus2": 3.06,
                "westeurope": 3.37, "northeurope": 3.33
            }}
        },
        "reserved_1yr_discount": 0.36,
        "reserved_3yr_discount": 0.56
    },
    "Lambda": {
        "provider_name": "Lambda Labs",
        "type": "cloud",
        "gpus": {
            "B200": {"instance": "gpu_8x_b200", "gpus_per_instance": 8, "price_per_gpu_hr": 5.74, "regions": {
                "us-west-1": 5.74, "us-south-1": 5.74
            }},
            "H100-SXM": {"instance": "gpu_8x_h100_sxm5", "gpus_per_instance": 8, "price_per_gpu_hr": 2.49, "regions": {
                "us-west-1": 2.49, "us-south-1": 2.49, "us-east-1": 2.49,
                "europe-central-1": 2.79
            }},
            "H100-PCIe": {"instance": "gpu_1x_h100_pcie", "gpus_per_instance": 1, "price_per_gpu_hr": 2.86, "regions": {
                "us-west-1": 2.86, "us-south-1": 2.86, "us-east-1": 2.86
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
        "reserved_1yr_discount": 0.20,
        "reserved_3yr_discount": 0.35
    },
    "CoreWeave": {
        "provider_name": "CoreWeave",
        "type": "cloud",
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
            "MI325X": {"instance": "mi325x-256gb", "gpus_per_instance": 1, "price_per_gpu_hr": 3.20, "regions": {
                "LAS1": 3.20, "ORD1": 3.20
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
        "reserved_1yr_discount": 0.25,
        "reserved_3yr_discount": 0.45
    },
    "RunPod": {
        "provider_name": "RunPod",
        "type": "marketplace",
        "gpus": {
            "B200": {"instance": "b200-sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 5.98, "regions": {
                "US": 5.98
            }},
            "H200": {"instance": "h200-sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 3.59, "regions": {
                "US": 3.59
            }},
            "H100-SXM": {"instance": "h100-sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 1.99, "regions": {
                "US": 1.99, "EU": 2.19
            }},
            "H100-PCIe": {"instance": "h100-pcie", "gpus_per_instance": 1, "price_per_gpu_hr": 1.79, "regions": {
                "US": 1.79, "EU": 1.99
            }},
            "MI325X": {"instance": "mi325x", "gpus_per_instance": 1, "price_per_gpu_hr": 2.99, "regions": {
                "US": 2.99, "EU": 3.29
            }},
            "MI300X": {"instance": "mi300x", "gpus_per_instance": 1, "price_per_gpu_hr": 2.49, "regions": {
                "US": 2.49, "EU": 2.69
            }},
            "A100-80GB": {"instance": "a100-80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 1.29, "regions": {
                "US": 1.29, "EU": 1.44
            }},
            "A100-40GB": {"instance": "a100-40gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.84, "regions": {
                "US": 0.84, "EU": 0.94
            }},
            "L40S": {"instance": "l40s", "gpus_per_instance": 1, "price_per_gpu_hr": 0.79, "regions": {
                "US": 0.79
            }},
            "RTX-4090": {"instance": "rtx4090", "gpus_per_instance": 1, "price_per_gpu_hr": 0.34, "regions": {
                "US": 0.34, "EU": 0.39
            }},
            "A10G": {"instance": "a10g", "gpus_per_instance": 1, "price_per_gpu_hr": 0.39, "regions": {
                "US": 0.39
            }}
        },
        "reserved_1yr_discount": 0.15,
        "reserved_3yr_discount": 0.30
    },
    "Vast.ai": {
        "provider_name": "Vast.ai",
        "type": "marketplace",
        "gpus": {
            "B200": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 2.67, "regions": {
                "US": 2.67, "EU": 3.10
            }},
            "H200": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.97, "regions": {
                "US": 1.97, "EU": 2.20, "APAC": 2.40
            }},
            "H100-SXM": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.70, "regions": {
                "US": 1.70, "EU": 1.90, "APAC": 2.05
            }},
            "H100-PCIe": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.55, "regions": {
                "US": 1.55, "EU": 1.70
            }},
            "MI300X": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 1.50, "regions": {
                "US": 1.50, "EU": 1.70
            }},
            "A100-80GB": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.85, "regions": {
                "US": 0.85, "EU": 1.00, "APAC": 1.10
            }},
            "A100-40GB": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.55, "regions": {
                "US": 0.55, "EU": 0.65
            }},
            "MI250X": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.65, "regions": {
                "US": 0.65, "EU": 0.75
            }},
            "RTX-4090": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.29, "regions": {
                "US": 0.29, "EU": 0.34, "APAC": 0.38
            }},
            "L40S": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.47, "regions": {
                "US": 0.47, "EU": 0.57
            }},
            "RTX-5090": {"instance": "community", "gpus_per_instance": 1, "price_per_gpu_hr": 0.45, "regions": {
                "US": 0.45, "EU": 0.55
            }}
        },
        "reserved_1yr_discount": 0.0,
        "reserved_3yr_discount": 0.0
    },
    "FluidStack": {
        "provider_name": "FluidStack",
        "type": "marketplace",
        "gpus": {
            "H200": {"instance": "h200_sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 2.30, "regions": {
                "US": 2.30, "EU": 2.53
            }},
            "H100-SXM": {"instance": "h100_sxm", "gpus_per_instance": 1, "price_per_gpu_hr": 2.10, "regions": {
                "US": 2.10, "EU": 2.31, "APAC": 2.45
            }},
            "MI300X": {"instance": "mi300x", "gpus_per_instance": 1, "price_per_gpu_hr": 1.75, "regions": {
                "US": 1.75, "EU": 1.93
            }},
            "A100-80GB": {"instance": "a100_80gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.95, "regions": {
                "US": 0.95, "EU": 1.10
            }},
            "A100-40GB": {"instance": "a100_40gb", "gpus_per_instance": 1, "price_per_gpu_hr": 0.65, "regions": {
                "US": 0.65
            }},
            "L40S": {"instance": "l40s", "gpus_per_instance": 1, "price_per_gpu_hr": 0.69, "regions": {
                "US": 0.69
            }}
        },
        "reserved_1yr_discount": 0.20,
        "reserved_3yr_discount": 0.35
    },
    "Oracle": {
        "provider_name": "Oracle Cloud (OCI)",
        "type": "cloud",
        "gpus": {
            "B200": {"instance": "BM.GPU.B200.8", "gpus_per_instance": 8, "price_per_gpu_hr": 4.25, "regions": {
                "us-ashburn-1": 4.25, "us-phoenix-1": 4.25,
                "uk-london-1": 4.68, "eu-frankfurt-1": 4.68
            }},
            "H200": {"instance": "BM.GPU.H200.8", "gpus_per_instance": 8, "price_per_gpu_hr": 10.00, "regions": {
                "us-ashburn-1": 10.00, "us-phoenix-1": 10.00,
                "uk-london-1": 11.00, "eu-frankfurt-1": 11.00
            }},
            "H100-SXM": {"instance": "BM.GPU.H100.8", "gpus_per_instance": 8, "price_per_gpu_hr": 3.19, "regions": {
                "us-ashburn-1": 3.19, "us-phoenix-1": 3.19, "uk-london-1": 3.51,
                "eu-frankfurt-1": 3.51, "ap-tokyo-1": 3.83
            }},
            "A100-80GB": {"instance": "BM.GPU.A100-v2.8", "gpus_per_instance": 8, "price_per_gpu_hr": 2.95, "regions": {
                "us-ashburn-1": 2.95, "us-phoenix-1": 2.95,
                "uk-london-1": 3.25, "eu-frankfurt-1": 3.25
            }},
            "A10G": {"instance": "VM.GPU.A10.1", "gpus_per_instance": 1, "price_per_gpu_hr": 0.70, "regions": {
                "us-ashburn-1": 0.70, "us-phoenix-1": 0.70
            }}
        },
        "reserved_1yr_discount": 0.30,
        "reserved_3yr_discount": 0.50
    },
    "Together": {
        "provider_name": "Together AI",
        "type": "cloud",
        "gpus": {
            "H100-SXM": {"instance": "dedicated", "gpus_per_instance": 1, "price_per_gpu_hr": 2.50, "regions": {
                "US": 2.50
            }},
            "A100-80GB": {"instance": "dedicated", "gpus_per_instance": 1, "price_per_gpu_hr": 1.50, "regions": {
                "US": 1.50
            }}
        },
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
        "2025-04": {"avg": 5.55, "min": 5.10, "max": 6.40, "availability": "scarce"},
        "2025-05": {"avg": 5.35, "min": 4.85, "max": 6.15, "availability": "scarce"},
        "2025-06": {"avg": 5.15, "min": 4.60, "max": 5.95, "availability": "scarce"},
        "2025-07": {"avg": 4.98, "min": 4.40, "max": 5.75, "availability": "limited"},
        "2025-08": {"avg": 4.82, "min": 4.25, "max": 5.60, "availability": "limited"},
        "2025-09": {"avg": 4.70, "min": 4.10, "max": 5.50, "availability": "limited"},
        "2025-10": {"avg": 4.85, "min": 4.20, "max": 5.70, "availability": "scarce"},
        "2025-11": {"avg": 4.95, "min": 4.30, "max": 5.85, "availability": "scarce"},
        "2025-12": {"avg": 5.10, "min": 4.40, "max": 6.00, "availability": "scarce"},
        "2026-01": {"avg": 5.05, "min": 4.35, "max": 5.95, "availability": "scarce"},
        "2026-02": {"avg": 4.98, "min": 4.30, "max": 5.85, "availability": "scarce"}
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
        "2024-10": {"avg": 2.65, "min": 2.10, "max": 3.50, "availability": "good"},
        "2024-11": {"avg": 2.55, "min": 2.02, "max": 3.40, "availability": "good"},
        "2024-12": {"avg": 2.48, "min": 1.95, "max": 3.30, "availability": "moderate"},
        "2025-01": {"avg": 2.42, "min": 1.90, "max": 3.25, "availability": "moderate"},
        "2025-02": {"avg": 2.38, "min": 1.88, "max": 3.20, "availability": "moderate"},
        "2025-03": {"avg": 2.32, "min": 1.85, "max": 3.15, "availability": "moderate"},
        "2025-04": {"avg": 2.28, "min": 1.82, "max": 3.10, "availability": "moderate"},
        "2025-05": {"avg": 2.22, "min": 1.78, "max": 3.05, "availability": "moderate"},
        "2025-06": {"avg": 2.15, "min": 1.72, "max": 2.95, "availability": "moderate"},
        "2025-07": {"avg": 2.10, "min": 1.68, "max": 2.88, "availability": "moderate"},
        "2025-08": {"avg": 2.05, "min": 1.62, "max": 2.82, "availability": "moderate"},
        "2025-09": {"avg": 2.00, "min": 1.55, "max": 2.78, "availability": "limited"},
        "2025-10": {"avg": 2.08, "min": 1.60, "max": 2.90, "availability": "limited"},
        "2025-11": {"avg": 2.15, "min": 1.65, "max": 3.00, "availability": "limited"},
        "2025-12": {"avg": 2.28, "min": 1.72, "max": 3.15, "availability": "limited"},
        "2026-01": {"avg": 2.35, "min": 1.78, "max": 3.25, "availability": "limited"},
        "2026-02": {"avg": 2.32, "min": 1.75, "max": 3.20, "availability": "limited"}
    },
    "H200": {
        "2024-04": {"avg": 5.20, "min": 4.70, "max": 6.20, "availability": "scarce"},
        "2024-05": {"avg": 5.05, "min": 4.55, "max": 6.05, "availability": "scarce"},
        "2024-06": {"avg": 4.90, "min": 4.40, "max": 5.90, "availability": "scarce"},
        "2024-07": {"avg": 4.70, "min": 4.10, "max": 5.70, "availability": "scarce"},
        "2024-08": {"avg": 4.50, "min": 3.90, "max": 5.50, "availability": "scarce"},
        "2024-09": {"avg": 4.35, "min": 3.75, "max": 5.30, "availability": "limited"},
        "2024-10": {"avg": 4.15, "min": 3.55, "max": 5.00, "availability": "limited"},
        "2024-11": {"avg": 4.00, "min": 3.40, "max": 4.80, "availability": "limited"},
        "2024-12": {"avg": 3.85, "min": 3.30, "max": 4.60, "availability": "limited"},
        "2025-01": {"avg": 3.72, "min": 3.20, "max": 4.45, "availability": "limited"},
        "2025-02": {"avg": 3.65, "min": 3.15, "max": 4.38, "availability": "limited"},
        "2025-03": {"avg": 3.58, "min": 3.10, "max": 4.30, "availability": "limited"},
        "2025-04": {"avg": 3.52, "min": 3.05, "max": 4.22, "availability": "limited"},
        "2025-05": {"avg": 3.48, "min": 3.00, "max": 4.18, "availability": "limited"},
        "2025-06": {"avg": 3.42, "min": 2.95, "max": 4.10, "availability": "limited"},
        "2025-07": {"avg": 3.38, "min": 2.92, "max": 4.05, "availability": "limited"},
        "2025-08": {"avg": 3.35, "min": 2.88, "max": 3.98, "availability": "limited"},
        "2025-09": {"avg": 3.40, "min": 2.92, "max": 4.10, "availability": "scarce"},
        "2025-10": {"avg": 3.52, "min": 3.00, "max": 4.25, "availability": "scarce"},
        "2025-11": {"avg": 3.60, "min": 3.08, "max": 4.35, "availability": "scarce"},
        "2025-12": {"avg": 3.68, "min": 3.15, "max": 4.45, "availability": "scarce"},
        "2026-01": {"avg": 3.72, "min": 3.20, "max": 4.50, "availability": "scarce"},
        "2026-02": {"avg": 3.70, "min": 3.18, "max": 4.48, "availability": "scarce"}
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
        "2025-01": {"avg": 1.34, "min": 1.00, "max": 1.92, "availability": "good"},
        "2025-02": {"avg": 1.30, "min": 0.96, "max": 1.88, "availability": "good"},
        "2025-03": {"avg": 1.28, "min": 0.94, "max": 1.85, "availability": "good"},
        "2025-04": {"avg": 1.25, "min": 0.92, "max": 1.82, "availability": "good"},
        "2025-05": {"avg": 1.22, "min": 0.90, "max": 1.78, "availability": "good"},
        "2025-06": {"avg": 1.18, "min": 0.88, "max": 1.72, "availability": "good"},
        "2025-07": {"avg": 1.15, "min": 0.85, "max": 1.68, "availability": "good"},
        "2025-08": {"avg": 1.12, "min": 0.82, "max": 1.65, "availability": "good"},
        "2025-09": {"avg": 1.10, "min": 0.80, "max": 1.62, "availability": "moderate"},
        "2025-10": {"avg": 1.15, "min": 0.84, "max": 1.70, "availability": "moderate"},
        "2025-11": {"avg": 1.20, "min": 0.88, "max": 1.78, "availability": "moderate"},
        "2025-12": {"avg": 1.28, "min": 0.92, "max": 1.88, "availability": "moderate"},
        "2026-01": {"avg": 1.30, "min": 0.95, "max": 1.90, "availability": "moderate"},
        "2026-02": {"avg": 1.28, "min": 0.92, "max": 1.85, "availability": "moderate"}
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
    },
    "GB200": {
        "2025-06": {"avg": 28.00, "min": 26.50, "max": 30.00, "availability": "scarce"},
        "2025-07": {"avg": 27.50, "min": 26.00, "max": 29.50, "availability": "scarce"},
        "2025-08": {"avg": 27.20, "min": 25.80, "max": 29.00, "availability": "scarce"},
        "2025-09": {"avg": 27.00, "min": 25.50, "max": 28.80, "availability": "scarce"},
        "2025-10": {"avg": 27.10, "min": 25.60, "max": 29.00, "availability": "scarce"},
        "2025-11": {"avg": 27.00, "min": 25.40, "max": 28.80, "availability": "scarce"},
        "2025-12": {"avg": 27.05, "min": 25.50, "max": 28.90, "availability": "scarce"},
        "2026-01": {"avg": 27.04, "min": 25.40, "max": 28.80, "availability": "scarce"},
        "2026-02": {"avg": 26.90, "min": 25.20, "max": 28.60, "availability": "scarce"}
    },
    "H100-PCIe": {
        "2023-06": {"avg": 3.20, "min": 2.90, "max": 3.60, "availability": "scarce"},
        "2023-09": {"avg": 3.10, "min": 2.80, "max": 3.50, "availability": "limited"},
        "2023-12": {"avg": 3.00, "min": 2.70, "max": 3.40, "availability": "limited"},
        "2024-03": {"avg": 2.90, "min": 2.60, "max": 3.30, "availability": "available"},
        "2024-06": {"avg": 2.80, "min": 2.50, "max": 3.20, "availability": "available"},
        "2024-09": {"avg": 2.70, "min": 2.40, "max": 3.10, "availability": "available"},
        "2024-12": {"avg": 2.60, "min": 2.30, "max": 3.00, "availability": "available"},
        "2025-03": {"avg": 2.40, "min": 2.10, "max": 2.80, "availability": "available"},
        "2025-06": {"avg": 2.20, "min": 1.95, "max": 2.55, "availability": "available"},
        "2025-09": {"avg": 2.05, "min": 1.80, "max": 2.40, "availability": "available"},
        "2025-12": {"avg": 1.95, "min": 1.70, "max": 2.25, "availability": "available"},
        "2026-02": {"avg": 1.90, "min": 1.65, "max": 2.20, "availability": "available"}
    },
    "A10G": {
        "2023-01": {"avg": 1.20, "min": 1.00, "max": 1.45, "availability": "available"},
        "2023-06": {"avg": 1.15, "min": 0.95, "max": 1.40, "availability": "available"},
        "2023-12": {"avg": 1.10, "min": 0.90, "max": 1.35, "availability": "available"},
        "2024-06": {"avg": 1.00, "min": 0.80, "max": 1.25, "availability": "available"},
        "2024-12": {"avg": 0.85, "min": 0.65, "max": 1.10, "availability": "abundant"},
        "2025-06": {"avg": 0.72, "min": 0.55, "max": 0.95, "availability": "abundant"},
        "2025-12": {"avg": 0.62, "min": 0.45, "max": 0.82, "availability": "abundant"},
        "2026-02": {"avg": 0.58, "min": 0.39, "max": 0.78, "availability": "abundant"}
    },
    "RTX-5090": {
        "2025-03": {"avg": 0.60, "min": 0.50, "max": 0.75, "availability": "scarce"},
        "2025-06": {"avg": 0.55, "min": 0.47, "max": 0.68, "availability": "limited"},
        "2025-09": {"avg": 0.50, "min": 0.44, "max": 0.62, "availability": "limited"},
        "2025-12": {"avg": 0.48, "min": 0.42, "max": 0.58, "availability": "available"},
        "2026-02": {"avg": 0.45, "min": 0.40, "max": 0.55, "availability": "available"}
    },
    "V100": {
        "2021-01": {"avg": 3.20, "min": 2.90, "max": 3.60, "availability": "available"},
        "2021-06": {"avg": 3.40, "min": 3.00, "max": 3.90, "availability": "limited"},
        "2022-01": {"avg": 3.50, "min": 3.10, "max": 4.00, "availability": "limited"},
        "2022-06": {"avg": 3.30, "min": 2.90, "max": 3.80, "availability": "available"},
        "2023-01": {"avg": 3.10, "min": 2.70, "max": 3.60, "availability": "available"},
        "2023-06": {"avg": 2.90, "min": 2.50, "max": 3.40, "availability": "available"},
        "2024-01": {"avg": 2.50, "min": 2.10, "max": 3.00, "availability": "abundant"},
        "2024-06": {"avg": 2.20, "min": 1.80, "max": 2.70, "availability": "abundant"},
        "2025-01": {"avg": 1.80, "min": 1.40, "max": 2.30, "availability": "abundant"},
        "2025-06": {"avg": 1.50, "min": 1.10, "max": 2.00, "availability": "abundant"},
        "2026-02": {"avg": 1.20, "min": 0.85, "max": 1.65, "availability": "abundant"}
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
    "flagship_lead_time_weeks": {"2023-01": 48, "2023-03": 48, "2023-06": 40, "2023-09": 36, "2023-12": 28,
                              "2024-03": 16, "2024-06": 10, "2024-09": 8, "2024-11": 52,
                              "2025-01": 48, "2025-03": 44, "2025-06": 40, "2025-09": 36, "2025-12": 36, "2026-01": 36, "2026-02": 36},
    "amd_gpu_market_share_pct": {"2023-01": 3, "2023-06": 5, "2024-01": 8, "2024-06": 12, "2025-01": 16, "2025-06": 19, "2026-01": 22},
    "gpu_lead_times": {
        "B200": {"weeks": 4, "status": "available", "note": "Shipping, broadly available"},
        "GB200": {"weeks": 10, "status": "limited", "note": "NVL72 racks ramping"},
        "H200": {"weeks": 12, "status": "limited", "note": "8-20 wk depending on volume"},
        "H100-SXM": {"weeks": 2, "status": "available", "note": "Broadly available"},
        "H100-PCIe": {"weeks": 2, "status": "available", "note": "Broadly available"},
        "A100-80GB": {"weeks": 1, "status": "available", "note": "Commodity"},
        "A100-40GB": {"weeks": 1, "status": "available", "note": "Commodity"},
        "MI300X": {"weeks": 10, "status": "limited", "note": "Ramping production"},
        "MI325X": {"weeks": 14, "status": "limited", "note": "Recently launched"},
        "L40S": {"weeks": 1, "status": "available", "note": "Broadly available"},
        "RTX-4090": {"weeks": 1, "status": "available", "note": "Consumer stock available"}
    }
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
        "adoption_trend_quarterly": {
            "2024-Q1": 19.5, "2024-Q2": 25.3, "2024-Q3": 33.8, "2024-Q4": 38.2,
            "2025-Q1": 42.5, "2025-Q2": 62.4, "2025-Q3": 72.8, "2025-Q4": 82.0,
            "2026-Q1": 88.5
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
        "adoption_trend_quarterly": {
            "2024-Q1": 2.3, "2024-Q2": 3.0, "2024-Q3": 4.0, "2024-Q4": 4.5,
            "2025-Q1": 4.3, "2025-Q2": 3.9, "2025-Q3": 4.2, "2025-Q4": 4.8,
            "2026-Q1": 5.2
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
        "adoption_trend_quarterly": {
            "2024-Q1": 4.3, "2024-Q2": 5.6, "2024-Q3": 7.5, "2024-Q4": 8.2,
            "2025-Q1": 6.3, "2025-Q2": 5.7, "2025-Q3": 6.5, "2025-Q4": 7.2,
            "2026-Q1": 7.8
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
        "adoption_trend_quarterly": {
            "2024-Q1": 0.5, "2024-Q2": 0.7, "2024-Q3": 0.9, "2024-Q4": 1.1,
            "2025-Q1": 1.3, "2025-Q2": 1.6, "2025-Q3": 2.0, "2025-Q4": 2.4,
            "2026-Q1": 2.8
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
        "adoption_trend_quarterly": {
            "2024-Q1": 0.4, "2024-Q2": 0.5, "2024-Q3": 0.7, "2024-Q4": 0.9,
            "2025-Q1": 1.0, "2025-Q2": 1.2, "2025-Q3": 1.5, "2025-Q4": 1.8,
            "2026-Q1": 2.1
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
        "adoption_trend_quarterly": {
            "2024-Q1": 6.6, "2024-Q2": 8.6, "2024-Q3": 11.5, "2024-Q4": 10.2,
            "2025-Q1": 9.5, "2025-Q2": 9.5, "2025-Q3": 10.2, "2025-Q4": 10.8,
            "2026-Q1": 11.2
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

# Per-GPU profiles: purchase price, cloud pricing tiers, self-hosted opex,
# and headline perf numbers — everything needed for TCO scenario comparison.
TCO_PROFILES = {
    "H100-SXM": {
        "gpu_price_usd": 30000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 2.49,
        "cloud_reserved_1yr_hr": 1.74,
        "cloud_reserved_3yr_hr": 1.12,
        "cloud_spot_hr": 1.70,
        "self_hosted": {
            "power_kwh": 0.58,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.12,
            "storage_hr": 0.08,
            "colo_rack_hr": 0.15,
            "ops_staff_hr": 0.10,
        },
        "perf_fp16_tflops": 989.5,
        "vram_gb": 80,
    },
    "B200": {
        "gpu_price_usd": 40000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 3.97,
        "cloud_reserved_1yr_hr": 2.78,
        "cloud_reserved_3yr_hr": 1.79,
        "cloud_spot_hr": 2.67,
        "self_hosted": {
            "power_kwh": 0.82,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.15,
            "storage_hr": 0.10,
            "colo_rack_hr": 0.18,
            "ops_staff_hr": 0.12,
        },
        "perf_fp16_tflops": 2250,
        "vram_gb": 192,
    },
    "H200": {
        "gpu_price_usd": 35000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 2.89,
        "cloud_reserved_1yr_hr": 2.02,
        "cloud_reserved_3yr_hr": 1.30,
        "cloud_spot_hr": 1.97,
        "self_hosted": {
            "power_kwh": 0.58,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.13,
            "storage_hr": 0.09,
            "colo_rack_hr": 0.16,
            "ops_staff_hr": 0.10,
        },
        "perf_fp16_tflops": 989.5,
        "vram_gb": 141,
    },
    "GB200": {
        "gpu_price_usd": 70000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 7.50,
        "cloud_reserved_1yr_hr": 5.25,
        "cloud_reserved_3yr_hr": 3.38,
        "cloud_spot_hr": 5.00,
        "self_hosted": {
            "power_kwh": 2.20,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.20,
            "storage_hr": 0.15,
            "colo_rack_hr": 0.25,
            "ops_staff_hr": 0.15,
        },
        "perf_fp16_tflops": 4500,
        "vram_gb": 384,
    },
    "A100-80GB": {
        "gpu_price_usd": 15000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 1.89,
        "cloud_reserved_1yr_hr": 1.32,
        "cloud_reserved_3yr_hr": 0.85,
        "cloud_spot_hr": 0.85,
        "self_hosted": {
            "power_kwh": 0.33,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.10,
            "storage_hr": 0.07,
            "colo_rack_hr": 0.12,
            "ops_staff_hr": 0.08,
        },
        "perf_fp16_tflops": 312,
        "vram_gb": 80,
    },
    "MI300X": {
        "gpu_price_usd": 15000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 2.19,
        "cloud_reserved_1yr_hr": 1.53,
        "cloud_reserved_3yr_hr": 0.99,
        "cloud_spot_hr": 1.50,
        "self_hosted": {
            "power_kwh": 0.62,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.12,
            "storage_hr": 0.08,
            "colo_rack_hr": 0.14,
            "ops_staff_hr": 0.10,
        },
        "perf_fp16_tflops": 1307,
        "vram_gb": 192,
    },
    "MI325X": {
        "gpu_price_usd": 20000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 3.20,
        "cloud_reserved_1yr_hr": 2.24,
        "cloud_reserved_3yr_hr": 1.44,
        "cloud_spot_hr": 2.10,
        "self_hosted": {
            "power_kwh": 0.83,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.13,
            "storage_hr": 0.09,
            "colo_rack_hr": 0.15,
            "ops_staff_hr": 0.10,
        },
        "perf_fp16_tflops": 1307.4,
        "vram_gb": 256,
    },
    "RTX-4090": {
        "gpu_price_usd": 1599,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 0.74,
        "cloud_reserved_1yr_hr": 0.52,
        "cloud_reserved_3yr_hr": 0.33,
        "cloud_spot_hr": 0.29,
        "self_hosted": {
            "power_kwh": 0.37,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.04,
            "storage_hr": 0.03,
            "colo_rack_hr": 0.06,
            "ops_staff_hr": 0.04,
        },
        "perf_fp16_tflops": 330,
        "vram_gb": 24,
    },
    "L40S": {
        "gpu_price_usd": 8000,
        "useful_life_years": 4,
        "cloud_on_demand_hr": 1.49,
        "cloud_reserved_1yr_hr": 1.04,
        "cloud_reserved_3yr_hr": 0.67,
        "cloud_spot_hr": 1.14,
        "self_hosted": {
            "power_kwh": 0.30,
            "power_cost_kwh": 0.08,
            "pue": 1.3,
            "networking_hr": 0.05,
            "storage_hr": 0.04,
            "colo_rack_hr": 0.08,
            "ops_staff_hr": 0.05,
        },
        "perf_fp16_tflops": 362,
        "vram_gb": 48,
    },
}

# Backward-compat alias — /api/tco endpoint still returns this
TCO_COMPONENTS = TCO_PROFILES

# ============================================================================
# INFERENCE ECONOMICS — $/M tokens by model and GPU
# ============================================================================

INFERENCE_BENCHMARKS = {
    # ── Top 20 models by weekly usage on OpenRouter (Feb 2026) ──
    # Ranking: openrouter.ai/rankings — top-weekly (paid models, excl free promos)
    # Pricing: $/M tokens (input/output) from OpenRouter + native APIs
    # tokens_7d: billions of tokens/week — Feb 16 chart data for top 9,
    #            remaining derived from "Others" bucket (~5641B) by API rank
    # context_k: context window in K tokens
    # open_source: True for OSS-weight models

    # #1 — MiniMax M2.5 (MiniMax) — chart: 2,571B
    "MiniMax-M2.5": {"params_b": 456, "type": "LLM", "category": "Frontier", "rank": 1,
        "tokens_7d": 2571.2, "context_k": 196, "open_source": True,
        "gpus": {},
        "providers": {"OpenRouter": {"input": 0.30, "output": 1.10}}
    },
    # #2 — Kimi K2.5 (MoonshotAI) — chart: 1,037B
    "Kimi-K2.5": {"params_b": 1000, "type": "LLM", "category": "Frontier", "rank": 2,
        "tokens_7d": 1036.7, "context_k": 262, "open_source": True,
        "gpus": {},
        "providers": {"OpenRouter": {"input": 0.45, "output": 2.20}}
    },
    # #3 — Gemini 3 Flash Preview (Google) — chart: 859B
    "Gemini-3-Flash": {"params_b": 80, "type": "LLM", "category": "Large", "rank": 3,
        "tokens_7d": 859.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Google AI Studio": {"input": 0.50, "output": 3.00}, "Google Vertex": {"input": 0.50, "output": 3.00}, "OpenRouter": {"input": 0.50, "output": 3.00}}
    },
    # #4 — GLM 5 (Z.ai) — chart: 803B
    "GLM-5": {"params_b": 600, "type": "LLM", "category": "Frontier", "rank": 4,
        "tokens_7d": 803.4, "context_k": 204, "open_source": True,
        "gpus": {},
        "providers": {"OpenRouter": {"input": 0.95, "output": 2.55}}
    },
    # #5 — DeepSeek V3.2 (DeepSeek) — chart: 745B
    "DeepSeek-V3.2": {"params_b": 685, "type": "LLM", "category": "Frontier", "rank": 5,
        "tokens_7d": 745.1, "context_k": 163, "open_source": True,
        "gpus": {"H100-SXM": {"tokens_per_sec": 42, "cost_per_1m_tokens": 1.60, "vram_gb": 180}, "B200": {"tokens_per_sec": 85, "cost_per_1m_tokens": 0.82, "vram_gb": 180}},
        "providers": {"DeepSeek API": {"input": 0.26, "output": 0.38}, "OpenRouter": {"input": 0.26, "output": 0.38}}
    },
    # #6 — Grok 4.1 Fast (xAI) — chart: 669B
    "Grok-4.1-Fast": {"params_b": 314, "type": "LLM", "category": "Frontier", "rank": 6,
        "tokens_7d": 669.0, "context_k": 2000, "open_source": False,
        "gpus": {},
        "providers": {"xAI API": {"input": 0.20, "output": 0.50}, "OpenRouter": {"input": 0.20, "output": 0.50}}
    },
    # #7 — Claude Opus 4.6 (Anthropic) — chart: 643B
    "Claude-Opus-4.6": {"params_b": 350, "type": "LLM", "category": "Frontier", "rank": 7,
        "tokens_7d": 643.1, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Anthropic API": {"input": 5.00, "output": 25.00}, "OpenRouter": {"input": 5.00, "output": 25.00}, "AWS Bedrock": {"input": 5.00, "output": 25.00}, "Google Vertex": {"input": 5.00, "output": 25.00}, "Azure": {"input": 5.00, "output": 25.00}}
    },
    # #8 — Claude Sonnet 4.5 (Anthropic) — chart: 534B
    "Claude-Sonnet-4.5": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 8,
        "tokens_7d": 534.3, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Anthropic API": {"input": 3.00, "output": 15.00}, "OpenRouter": {"input": 3.00, "output": 15.00}, "AWS Bedrock": {"input": 3.00, "output": 15.00}, "Google Vertex": {"input": 3.00, "output": 15.00}, "Azure": {"input": 3.00, "output": 15.00}}
    },
    # #9 — Gemini 2.5 Flash (Google) — in "Others"; prev week chart: 451B
    "Gemini-2.5-Flash": {"params_b": 65, "type": "LLM", "category": "Large", "rank": 9,
        "tokens_7d": 451.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Google AI Studio": {"input": 0.15, "output": 0.60}, "Google Vertex": {"input": 0.30, "output": 2.50}, "OpenRouter": {"input": 0.30, "output": 2.50}}
    },
    # #10 — Gemini 2.5 Flash Lite (Google) — prev weeks chart: 344B
    "Gemini-2.5-Flash-Lite": {"params_b": 30, "type": "LLM", "category": "Medium", "rank": 10,
        "tokens_7d": 344.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Google AI Studio": {"input": 0.05, "output": 0.20}, "Google Vertex": {"input": 0.10, "output": 0.40}, "OpenRouter": {"input": 0.10, "output": 0.40}}
    },
    # #11 — GPT-5 Nano (OpenAI) — in "Others"
    "GPT-5-Nano": {"params_b": 20, "type": "LLM", "category": "Medium", "rank": 11,
        "tokens_7d": 310.0, "context_k": 400, "open_source": False,
        "gpus": {},
        "providers": {"OpenAI API": {"input": 0.05, "output": 0.40}, "OpenRouter": {"input": 0.05, "output": 0.40}, "Azure": {"input": 0.05, "output": 0.40}}
    },
    # #12 — Claude Sonnet 4.6 (Anthropic) — launched Feb 17; chart partial: 46B
    "Claude-Sonnet-4.6": {"params_b": 175, "type": "LLM", "category": "Frontier", "rank": 12,
        "tokens_7d": 280.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Anthropic API": {"input": 3.00, "output": 15.00}, "OpenRouter": {"input": 3.00, "output": 15.00}, "AWS Bedrock": {"input": 3.00, "output": 15.00}, "Google Vertex": {"input": 3.00, "output": 15.00}, "Azure": {"input": 3.00, "output": 15.00}}
    },
    # #13 — GPT-OSS-120B (OpenAI) — prev weeks chart: 299B
    "GPT-OSS-120B": {"params_b": 120, "type": "LLM", "category": "Large", "rank": 13,
        "tokens_7d": 299.0, "context_k": 131, "open_source": True,
        "gpus": {},
        "providers": {"OpenRouter": {"input": 0.039, "output": 0.19}}
    },
    # #14 — GPT-5.2 (OpenAI) — in "Others"
    "GPT-5.2": {"params_b": 500, "type": "LLM", "category": "Frontier", "rank": 14,
        "tokens_7d": 250.0, "context_k": 400, "open_source": False,
        "gpus": {},
        "providers": {"OpenAI API": {"input": 1.75, "output": 14.00}, "OpenRouter": {"input": 1.75, "output": 14.00}, "Azure": {"input": 1.75, "output": 14.00}}
    },
    # #15 — Gemini 2.0 Flash (Google) — prev weeks chart: 185B
    "Gemini-2.0-Flash": {"params_b": 50, "type": "LLM", "category": "Large", "rank": 15,
        "tokens_7d": 185.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Google AI Studio": {"input": 0.10, "output": 0.40}, "Google Vertex": {"input": 0.10, "output": 0.40}, "OpenRouter": {"input": 0.10, "output": 0.40}}
    },
    # #16 — Claude Opus 4.5 (Anthropic) — prev weeks chart: 370B
    "Claude-Opus-4.5": {"params_b": 350, "type": "LLM", "category": "Frontier", "rank": 16,
        "tokens_7d": 370.0, "context_k": 200, "open_source": False,
        "gpus": {},
        "providers": {"Anthropic API": {"input": 5.00, "output": 25.00}, "OpenRouter": {"input": 5.00, "output": 25.00}, "AWS Bedrock": {"input": 5.00, "output": 25.00}, "Google Vertex": {"input": 5.00, "output": 25.00}, "Azure": {"input": 5.00, "output": 25.00}}
    },
    # #17 — Gemini 3 Pro Preview (Google) — prev weeks chart: 170B
    "Gemini-3-Pro": {"params_b": 300, "type": "LLM", "category": "Frontier", "rank": 17,
        "tokens_7d": 170.0, "context_k": 1000, "open_source": False,
        "gpus": {},
        "providers": {"Google AI Studio": {"input": 2.00, "output": 12.00}, "Google Vertex": {"input": 2.00, "output": 12.00}, "OpenRouter": {"input": 2.00, "output": 12.00}}
    },
    # #18 — Claude Haiku 4.5 (Anthropic) — in "Others"
    "Claude-Haiku-4.5": {"params_b": 20, "type": "LLM", "category": "Medium", "rank": 18,
        "tokens_7d": 150.0, "context_k": 200, "open_source": False,
        "gpus": {},
        "providers": {"Anthropic API": {"input": 1.00, "output": 5.00}, "OpenRouter": {"input": 1.00, "output": 5.00}, "AWS Bedrock": {"input": 1.00, "output": 5.00}, "Google Vertex": {"input": 1.00, "output": 5.00}}
    },
    # #19 — GLM 4.7 (Z.ai) — in "Others"
    "GLM-4.7": {"params_b": 230, "type": "LLM", "category": "Frontier", "rank": 19,
        "tokens_7d": 130.0, "context_k": 202, "open_source": True,
        "gpus": {},
        "providers": {"OpenRouter": {"input": 0.38, "output": 1.70}}
    },
    # #20 — GPT-4o Mini (OpenAI) — prev weeks chart: 53B (declining)
    "GPT-4o-Mini": {"params_b": 70, "type": "LLM", "category": "Large", "rank": 20,
        "tokens_7d": 53.0, "context_k": 128, "open_source": False,
        "gpus": {},
        "providers": {"OpenAI API": {"input": 0.15, "output": 0.60}, "OpenRouter": {"input": 0.15, "output": 0.60}, "Azure": {"input": 0.15, "output": 0.60}}
    }
}

# ============================================================================
# PRICING TIERS — On-Demand vs Reserved across providers
# ============================================================================

SPOT_MARKET = {
    "B200": {
        "on_demand_low": 2.67, "on_demand_avg": 6.50, "on_demand_high": 14.24,
        "reserved_1yr_low": 2.00, "reserved_1yr_avg": 4.55, "reserved_1yr_high": 9.97,
        "reserved_3yr_low": 1.47, "reserved_3yr_avg": 3.25, "reserved_3yr_high": 6.41,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 6, "quarterly_trend": [12.80, 11.50, 10.20, 8.50]
    },
    "H200": {
        "on_demand_low": 1.97, "on_demand_avg": 5.30, "on_demand_high": 10.85,
        "reserved_1yr_low": 1.48, "reserved_1yr_avg": 3.71, "reserved_1yr_high": 7.60,
        "reserved_3yr_low": 1.08, "reserved_3yr_avg": 2.65, "reserved_3yr_high": 4.88,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 8, "quarterly_trend": [9.80, 8.60, 7.90, 7.60]
    },
    "H100-SXM": {
        "on_demand_low": 1.70, "on_demand_avg": 5.80, "on_demand_high": 12.75,
        "reserved_1yr_low": 1.28, "reserved_1yr_avg": 4.06, "reserved_1yr_high": 8.93,
        "reserved_3yr_low": 0.94, "reserved_3yr_avg": 2.90, "reserved_3yr_high": 5.74,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 10, "quarterly_trend": [8.20, 7.10, 6.40, 5.80]
    },
    "A100-80GB": {
        "on_demand_low": 0.85, "on_demand_avg": 2.60, "on_demand_high": 5.63,
        "reserved_1yr_low": 0.64, "reserved_1yr_avg": 1.82, "reserved_1yr_high": 3.94,
        "reserved_3yr_low": 0.47, "reserved_3yr_avg": 1.30, "reserved_3yr_high": 2.53,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 9, "quarterly_trend": [3.40, 3.10, 2.80, 2.60]
    },
    "MI300X": {
        "on_demand_low": 1.50, "on_demand_avg": 3.10, "on_demand_high": 6.00,
        "reserved_1yr_low": 1.13, "reserved_1yr_avg": 2.17, "reserved_1yr_high": 4.20,
        "reserved_3yr_low": 0.83, "reserved_3yr_avg": 1.55, "reserved_3yr_high": 2.70,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 5, "quarterly_trend": [4.20, 3.80, 3.40, 3.10]
    },
    "RTX-4090": {
        "on_demand_low": 0.29, "on_demand_avg": 0.45, "on_demand_high": 0.74,
        "reserved_1yr_low": 0.22, "reserved_1yr_avg": 0.34, "reserved_1yr_high": 0.56,
        "reserved_3yr_low": 0.16, "reserved_3yr_avg": 0.25, "reserved_3yr_high": 0.37,
        "res1_savings_pct": 25, "res3_savings_pct": 45,
        "num_providers": 4, "quarterly_trend": [0.65, 0.55, 0.48, 0.45]
    },
    "MI325X": {
        "on_demand_low": 2.99, "on_demand_avg": 5.80, "on_demand_high": 8.40,
        "reserved_1yr_low": 2.09, "reserved_1yr_avg": 4.06, "reserved_1yr_high": 5.88,
        "reserved_3yr_low": 1.50, "reserved_3yr_avg": 2.90, "reserved_3yr_high": 4.20,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 3, "quarterly_trend": [7.50, 6.80, 6.20, 5.80]
    },
    "L40S": {
        "on_demand_low": 0.47, "on_demand_avg": 1.00, "on_demand_high": 1.86,
        "reserved_1yr_low": 0.35, "reserved_1yr_avg": 0.70, "reserved_1yr_high": 1.30,
        "reserved_3yr_low": 0.26, "reserved_3yr_avg": 0.50, "reserved_3yr_high": 0.84,
        "res1_savings_pct": 30, "res3_savings_pct": 50,
        "num_providers": 6, "quarterly_trend": [1.45, 1.30, 1.18, 1.10]
    }
}

# ============================================================================
# NEWS & MARKET SIGNALS
# ============================================================================

COMMUNITY_SENTIMENT = {
    "H100-SXM": {
        "score": 88, "reddit_sentiment": 0.82, "github_compat_score": 95,
        "hf_models_trained": 48200, "mentions_30d": 32400,
        "ecosystem": "mature", "adoption": "stable", "community_pick": True,
        "pros": ["Universal framework support", "Massive ecosystem", "Proven at scale"],
        "cons": ["Price premium vs AMD"],
        "top_use_case": "LLM training & serving at scale"
    },
    "B200": {
        "score": 82, "reddit_sentiment": 0.78, "github_compat_score": 80,
        "hf_models_trained": 3100, "mentions_30d": 28600,
        "ecosystem": "growing", "adoption": "rising", "community_pick": False,
        "pros": ["2x H100 perf/watt", "192GB HBM3e", "FP4 support"],
        "cons": ["Still ramping ecosystem"],
        "top_use_case": "Next-gen LLM training & large batch inference"
    },
    "H200": {
        "score": 85, "reddit_sentiment": 0.80, "github_compat_score": 92,
        "hf_models_trained": 12500, "mentions_30d": 18200,
        "ecosystem": "mature", "adoption": "rising", "community_pick": False,
        "pros": ["141GB HBM3e", "H100-compatible", "Great for large models"],
        "cons": ["Being leapfrogged by B200"],
        "top_use_case": "Large model inference & fine-tuning"
    },
    "A100-80GB": {
        "score": 79, "reddit_sentiment": 0.75, "github_compat_score": 98,
        "hf_models_trained": 85400, "mentions_30d": 14100,
        "ecosystem": "mature", "adoption": "stable", "community_pick": False,
        "pros": ["Best price/perf for training", "Universal support", "Abundant supply"],
        "cons": ["Aging architecture"],
        "top_use_case": "Cost-effective fine-tuning & mid-size training"
    },
    "MI300X": {
        "score": 62, "reddit_sentiment": 0.58, "github_compat_score": 65,
        "hf_models_trained": 2800, "mentions_30d": 9400,
        "ecosystem": "growing", "adoption": "rising", "community_pick": False,
        "pros": ["192GB HBM3", "Competitive pricing", "ROCm improving"],
        "cons": ["ROCm gaps vs CUDA", "Smaller ecosystem"],
        "top_use_case": "Inference workloads & cost-optimized serving"
    },
    "L40S": {
        "score": 72, "reddit_sentiment": 0.70, "github_compat_score": 90,
        "hf_models_trained": 8200, "mentions_30d": 6800,
        "ecosystem": "mature", "adoption": "stable", "community_pick": False,
        "pros": ["Great for inference", "Low power", "Affordable"],
        "cons": ["No NVLink", "48GB VRAM limit"],
        "top_use_case": "Inference serving & fine-tuning small models"
    },
    "RTX-4090": {
        "score": 74, "reddit_sentiment": 0.85, "github_compat_score": 88,
        "hf_models_trained": 22100, "mentions_30d": 11200,
        "ecosystem": "mature", "adoption": "stable", "community_pick": True,
        "pros": ["Cheapest per hour", "Great for prototyping", "Consumer availability"],
        "cons": ["24GB VRAM limit", "Not for production"],
        "top_use_case": "Research prototyping & personal inference"
    }
}

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
                "provider_type": data.get("type", "cloud"),
                "instance": gpu["instance"],
                "price_per_gpu_hr": gpu["price_per_gpu_hr"],
                "price_monthly": gpu["price_per_gpu_hr"] * 730,
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
            "cheapest_provider_type": providers[0].get("provider_type", "cloud") if providers else "cloud",
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
        "comparison_matrix": comparison,
        "market_sentiment": COMMUNITY_SENTIMENT
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
    "A100-80GB": {"tdp_watts": 400, "typical_watts": 330, "kwh_per_hour": 0.33, "annual_kwh_full_util": 2891, "carbon_kg_per_year_us_avg": 1156, "carbon_kg_per_year_eu_nordic": 130, "water_liters_per_year_us_avg": 5204, "embodied_carbon_kg": 100},
    "MI300X": {"tdp_watts": 750, "typical_watts": 620, "kwh_per_hour": 0.62, "annual_kwh_full_util": 5431, "carbon_kg_per_year_us_avg": 2172, "carbon_kg_per_year_eu_nordic": 244, "water_liters_per_year_us_avg": 9776, "embodied_carbon_kg": 160},
    "H200": {"tdp_watts": 700, "typical_watts": 580, "kwh_per_hour": 0.58, "annual_kwh_full_util": 5081, "carbon_kg_per_year_us_avg": 2032, "carbon_kg_per_year_eu_nordic": 228, "water_liters_per_year_us_avg": 9146, "embodied_carbon_kg": 155},
    "RTX-4090": {"tdp_watts": 450, "typical_watts": 370, "kwh_per_hour": 0.37, "annual_kwh_full_util": 3241, "carbon_kg_per_year_us_avg": 1296, "carbon_kg_per_year_eu_nordic": 146, "water_liters_per_year_us_avg": 5834, "embodied_carbon_kg": 80},
    "MI325X": {"tdp_watts": 1000, "typical_watts": 830, "kwh_per_hour": 0.83, "annual_kwh_full_util": 7273, "carbon_kg_per_year_us_avg": 2909, "carbon_kg_per_year_eu_nordic": 327, "water_liters_per_year_us_avg": 13091, "embodied_carbon_kg": 175},
    "GB200": {"tdp_watts": 2700, "typical_watts": 2200, "kwh_per_hour": 2.20, "annual_kwh_full_util": 19272, "carbon_kg_per_year_us_avg": 7709, "carbon_kg_per_year_eu_nordic": 867, "water_liters_per_year_us_avg": 34690, "embodied_carbon_kg": 350}
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
    {
        "date": "2022-08", "regulation": "CHIPS and Science Act",
        "category": "Industrial Policy", "target": "US domestic",
        "status": "enacted", "impact": "high",
        "affected_gpus": [],
        "description": "$39B manufacturing subsidies + $13B R&D. Recipients barred from expanding chip fabs in China for 10 years. 25% investment tax credit for US fab construction.",
        "regional_impact": {"US": "major_benefit", "EU": "neutral", "China": "negative", "Japan": "neutral", "India": "positive", "Middle_East": "neutral", "SE_Asia": "positive"}
    },
    {
        "date": "2022-10", "regulation": "BIS Advanced Computing Export Controls",
        "category": "Export Control", "target": "China, Macau",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["A100", "H100", "MI250X"],
        "description": "First sweeping controls blocking A100-class+ chips to China. Introduced TPP and performance-density thresholds. Also restricted semiconductor manufacturing equipment.",
        "regional_impact": {"US": "mixed", "EU": "neutral", "China": "high_negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2023-07", "regulation": "Japan Semiconductor Equipment Controls",
        "category": "Export Control (Equipment)", "target": "China (de facto)",
        "status": "enacted", "impact": "high",
        "affected_gpus": [],
        "description": "Japan restricted 23 types of advanced chip manufacturing equipment (lithography, etching, deposition). Expanded by 21 items in Apr 2024. China stockpiled $5B in equipment pre-deadline.",
        "regional_impact": {"US": "positive", "EU": "neutral", "China": "high_negative", "Japan": "negative_domestic", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2023-07", "regulation": "China Gallium/Germanium Export Controls",
        "category": "Retaliation", "target": "Global (primarily US, EU, Japan)",
        "status": "enacted", "impact": "medium",
        "affected_gpus": [],
        "description": "China imposed export licenses on gallium and germanium. Escalated to graphite (Oct 2023), antimony (Aug 2024), tungsten (Feb 2025), rare earths (Apr 2025). Full US halt Dec 2024. Partially suspended Nov 2025 under trade truce.",
        "regional_impact": {"US": "negative", "EU": "negative", "China": "mixed", "Japan": "negative", "India": "positive", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2023-09", "regulation": "Netherlands ASML DUV Export Controls",
        "category": "Export Control (Equipment)", "target": "China (de facto)",
        "status": "enacted", "impact": "high",
        "affected_gpus": [],
        "description": "Blocked ASML from selling advanced DUV immersion lithography to China. Expanded Sep 2024 to require licenses for 1970i/1980i DUV machines. ASML China revenue projected to drop from 29% to ~20%.",
        "regional_impact": {"US": "positive", "EU": "mixed", "China": "high_negative", "Japan": "positive", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2023-10", "regulation": "BIS Updated Export Controls (Loophole Closure)",
        "category": "Export Control", "target": "China, Macau",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["A800", "H800", "RTX 4090", "MI300X", "L40S"],
        "description": "Closed loopholes — blocked A800/H800 China workaround variants. Added performance-density thresholds capturing RTX 4090 gaming GPU. Expanded Entity List.",
        "regional_impact": {"US": "mixed", "EU": "neutral", "China": "high_negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "low_negative"}
    },
    {
        "date": "2024-01", "regulation": "NVIDIA China Compliance SKUs (H20/L20)",
        "category": "Industry Response", "target": "China",
        "status": "enacted", "impact": "medium",
        "affected_gpus": ["H20", "L20"],
        "description": "NVIDIA released downgraded H20 and L20 chips designed to comply with US thresholds for the China market. Reduced memory bandwidth and compute performance vs H100.",
        "regional_impact": {"US": "neutral", "EU": "neutral", "China": "positive", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2024-08", "regulation": "EU AI Act (Entry into Force)",
        "category": "AI Governance", "target": "EU + any entity selling AI in EU",
        "status": "enacted", "impact": "high",
        "affected_gpus": [],
        "description": "World's first comprehensive AI law. Risk-based framework: prohibited/high/limited/minimal risk tiers. Penalties up to EUR 35M or 7% global turnover. Phase 1 (prohibited practices) Feb 2025, Phase 2 (GPAI) Aug 2025, Phase 3 (high-risk) Aug 2026.",
        "regional_impact": {"US": "negative", "EU": "mixed", "China": "low_negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2024-12", "regulation": "Biden Entity List + HBM Controls + FDPR Expansion",
        "category": "Export Control", "target": "China, Macau",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["HBM3/HBM3e stacks"],
        "description": "Added 140 entities to Entity List (fabs, tool makers, Huawei suppliers). First country-wide HBM export controls to China. Expanded Foreign Direct Product Rule scope. Restricted advanced AI chip packaging tech.",
        "regional_impact": {"US": "mixed", "EU": "low_negative", "China": "high_negative", "Japan": "low_negative", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "low_negative"}
    },
    {
        "date": "2025-01", "regulation": "Biden AI Diffusion Rule (3-Tier Framework)",
        "category": "Export Control", "target": "Global — 3 tiers",
        "status": "rescinded", "impact": "high",
        "affected_gpus": ["H100", "H200", "B100", "B200", "GB200", "MI300X", "MI325X"],
        "description": "3-tier global system: Tier 1 (18 allies) unrestricted; Tier 2 (~150 countries) quantity caps ~50K GPUs/entity; Tier 3 (~25 countries incl. China) prohibited. Also controlled AI model weights. Rescinded by Trump admin May 2025 before compliance date.",
        "regional_impact": {"US": "mixed", "EU": "positive", "China": "high_negative", "Japan": "positive", "India": "negative", "Middle_East": "negative", "SE_Asia": "negative"}
    },
    {
        "date": "2025-02", "regulation": "EU AI Act Phase 1 — Prohibited Practices",
        "category": "AI Governance", "target": "EU",
        "status": "in_effect", "impact": "medium",
        "affected_gpus": [],
        "description": "Ban on social scoring, emotion recognition in workplaces/schools, untargeted facial scraping, predictive crime assessment. First enforcement phase of EU AI Act.",
        "regional_impact": {"US": "low_negative", "EU": "mixed", "China": "neutral", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-05", "regulation": "Trump Rescinds AI Diffusion Rule",
        "category": "Deregulation", "target": "Global",
        "status": "enacted", "impact": "high",
        "affected_gpus": [],
        "description": "Eliminated Biden's 3-tier country system and quantity caps 2 days before compliance date. Replaced with bilateral deal-based approach. Pre-existing China entity-level controls remain.",
        "regional_impact": {"US": "positive", "EU": "neutral", "China": "neutral", "Japan": "neutral", "India": "positive", "Middle_East": "positive", "SE_Asia": "positive"}
    },
    {
        "date": "2025-05", "regulation": "BIS Huawei Ascend Worldwide Ban (GP10)",
        "category": "Export Control", "target": "Worldwide",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["Ascend 910B", "Ascend 910C", "Ascend 910D"],
        "description": "Declared using/selling/servicing Huawei Ascend 910B/C/D chips anywhere violates US export controls under Foreign Direct Product Rule. Extraterritorial worldwide scope.",
        "regional_impact": {"US": "positive", "EU": "low_negative", "China": "high_negative", "Japan": "neutral", "India": "low_negative", "Middle_East": "negative", "SE_Asia": "negative"}
    },
    {
        "date": "2025-07", "regulation": "Proposed Malaysia/Thailand Chip Restrictions",
        "category": "Export Control", "target": "Malaysia, Thailand",
        "status": "proposed", "impact": "medium",
        "affected_gpus": ["All advanced AI GPUs"],
        "description": "Draft BIS rule requiring export licenses for AI GPU shipments to Malaysia/Thailand — evidence of transshipment to China. Enhanced documentation and end-use verification.",
        "regional_impact": {"US": "neutral", "EU": "neutral", "China": "negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "high_negative"}
    },
    {
        "date": "2025-08", "regulation": "Trump H20 Revenue-Sharing Deal (15%)",
        "category": "Export License", "target": "China",
        "status": "enacted", "impact": "medium",
        "affected_gpus": ["H20"],
        "description": "Export license for NVIDIA/AMD to sell H20-class chips to China with 15% of China sales revenue paid to US government. First-ever revenue-sharing arrangement tied to export licenses.",
        "regional_impact": {"US": "mixed", "EU": "neutral", "China": "positive", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-08", "regulation": "EU AI Act Phase 2 — GPAI Obligations",
        "category": "AI Governance", "target": "EU",
        "status": "in_effect", "impact": "medium",
        "affected_gpus": [],
        "description": "General-Purpose AI model obligations take effect. AI Office becomes operational. Compute reporting requirements for GPAI providers. Governance infrastructure established.",
        "regional_impact": {"US": "negative", "EU": "mixed", "China": "low_negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-11", "regulation": "UAE/Saudi 70K GB300 Chip Authorization",
        "category": "Export License", "target": "UAE, Saudi Arabia",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["GB300"],
        "description": "Commerce authorized 70,000 NVIDIA GB300 chips to HUMAIN (Saudi) and G42 (UAE). Ended regulatory standoff blocking billions in data center investments. Part of US-Gulf AI partnership.",
        "regional_impact": {"US": "positive", "EU": "neutral", "China": "negative", "Japan": "neutral", "India": "low_negative", "Middle_East": "high_positive", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-11", "regulation": "China Rare Earth Export Truce",
        "category": "Trade Truce", "target": "US, Global",
        "status": "temporary", "impact": "medium",
        "affected_gpus": [],
        "description": "China suspended export controls on gallium, germanium, antimony, graphite, and rare earths for 1 year under trade truce. General licenses issued for US end users. Can be reimposed at any time.",
        "regional_impact": {"US": "positive", "EU": "positive", "China": "mixed", "Japan": "positive", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-12", "regulation": "Trump H200 to China (25% Revenue Share)",
        "category": "Export License", "target": "China (approved customers)",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["H200", "MI325X"],
        "description": "Authorized NVIDIA H200 sales to vetted Chinese customers with 25% revenue share to US government (up from 15% for H20). Near-cutting-edge chips to China under commercial terms. Bipartisan criticism.",
        "regional_impact": {"US": "mixed", "EU": "neutral", "China": "positive", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2025-12", "regulation": "Vietnam AI Law",
        "category": "AI Governance", "target": "Vietnam",
        "status": "enacted", "impact": "low",
        "affected_gpus": [],
        "description": "First SE Asian comprehensive AI law. Phased implementation starting Mar 2026 over 4 years. Establishes AI governance framework for Vietnamese market.",
        "regional_impact": {"US": "neutral", "EU": "neutral", "China": "neutral", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "mixed"}
    },
    {
        "date": "2026-01", "regulation": "BIS Codified H200/MI325X License Policy",
        "category": "Export Control", "target": "China, Macau",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["H200", "MI325X"],
        "description": "Codified shift from 'presumption of denial' to 'case-by-case review' for H200/MI325X-class exports to China. Requires: sufficient US domestic supply, no foundry capacity diversion, third-party testing, and recipient security procedures.",
        "regional_impact": {"US": "mixed", "EU": "neutral", "China": "positive", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2026-01", "regulation": "Section 232 Semiconductor Tariff (25%)",
        "category": "Tariff", "target": "All non-US origins",
        "status": "enacted", "impact": "high",
        "affected_gpus": ["All imported AI chips"],
        "description": "25% tariff on imported advanced AI semiconductors, SME, and derivative products. Broad exemptions for US data center imports, R&D, startups, consumer use, and public sector. Impact review due Jul 2026.",
        "regional_impact": {"US": "mixed", "EU": "negative", "China": "negative", "Japan": "negative", "India": "low_negative", "Middle_East": "neutral", "SE_Asia": "negative"}
    },
    {
        "date": "2026-01", "regulation": "AI OVERWATCH Act (H.R. 6875)",
        "category": "Legislative", "target": "Countries of concern (China)",
        "status": "in_committee", "impact": "high",
        "affected_gpus": ["All export-licensed AI GPUs"],
        "description": "Congressional oversight of AI chip exports like arms sales. Creates 'Trusted US Person' framework for ally deployments. Would temporarily revoke existing China licenses pending national security strategy review. Passed committee markup Jan 2026.",
        "regional_impact": {"US": "mixed", "EU": "positive", "China": "high_negative", "Japan": "positive", "India": "neutral", "Middle_East": "uncertain", "SE_Asia": "neutral"}
    },
    {
        "date": "2026-02", "regulation": "India Semiconductor Mission 2.0",
        "category": "Industrial Policy", "target": "India domestic",
        "status": "enacted", "impact": "medium",
        "affected_gpus": [],
        "description": "ISM 2.0 expanded beyond fab subsidies to semiconductor equipment, materials, and IP. 10 approved fab projects; $38B domestic market targeting $100B by 2030. Government subsidizes AI compute at Rs 65/GPU-hr vs global $2-3/hr.",
        "regional_impact": {"US": "positive", "EU": "neutral", "China": "low_negative", "Japan": "positive", "India": "high_positive", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
    {
        "date": "2026-08", "regulation": "EU AI Act Phase 3 — High-Risk AI Systems",
        "category": "AI Governance", "target": "EU",
        "status": "upcoming", "impact": "high",
        "affected_gpus": [],
        "description": "Full compliance for high-risk standalone AI systems: biometrics, critical infrastructure, employment, law enforcement, education, migration. Conformity assessments and CE marking required.",
        "regional_impact": {"US": "negative", "EU": "mixed", "China": "low_negative", "Japan": "neutral", "India": "neutral", "Middle_East": "neutral", "SE_Asia": "neutral"}
    },
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
            "RTX-4090": {"optimal_config": "2x RTX-4090", "batch_size": 32, "throughput_tok_s": 95, "cost_per_1m_tokens": 0.058, "vram_headroom_pct": 46, "fit_score": 72, "notes": "Best cost/perf if 2-GPU setup acceptable"}
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
            "H200": {"optimal_config": "2x H200", "batch_size": 64, "throughput_tok_s": 110, "cost_per_1m_tokens": 0.42, "vram_headroom_pct": 50, "fit_score": 80, "notes": "NVLink pair, good throughput"}
        }
    },
    "180B": {
        "models": ["Falcon-180B", "DBRX-132B"],
        "vram_required_gb": 360,
        "gpus": {
            "H100-SXM": {"optimal_config": "8x H100 (DGX)", "batch_size": 32, "throughput_tok_s": 48, "cost_per_1m_tokens": 1.20, "vram_headroom_pct": 44, "fit_score": 75, "notes": "Full DGX node, 640GB provides ample headroom for batching"},
            "B200": {"optimal_config": "2x B200", "batch_size": 64, "throughput_tok_s": 85, "cost_per_1m_tokens": 0.82, "vram_headroom_pct": 6, "fit_score": 85, "notes": "384GB fits 180B FP16 comfortably"},
            "MI300X": {"optimal_config": "2x MI300X", "batch_size": 48, "throughput_tok_s": 55, "cost_per_1m_tokens": 0.95, "vram_headroom_pct": 6, "fit_score": 80, "notes": "384GB HBM, good AMD value"},
            "GB200": {"optimal_config": "1x GB200", "batch_size": 128, "throughput_tok_s": 120, "cost_per_1m_tokens": 0.60, "vram_headroom_pct": 6, "fit_score": 95, "notes": "Single NVL72 node fits 180B with headroom"},
            "H200": {"optimal_config": "4x H200", "batch_size": 32, "throughput_tok_s": 58, "cost_per_1m_tokens": 1.10, "vram_headroom_pct": 36, "fit_score": 78, "notes": "564GB total, good headroom for batching"}
        }
    },
    "405B": {
        "models": ["Llama-3.1-405B"],
        "vram_required_gb": 810,
        "gpus": {
            "H100-SXM": {"optimal_config": "16x H100 (2x DGX)", "batch_size": 8, "throughput_tok_s": 28, "cost_per_1m_tokens": 2.80, "vram_headroom_pct": 37, "fit_score": 55, "notes": "Needs 2 DGX nodes (1.28TB), cross-node NVLink"},
            "B200": {"optimal_config": "8x B200 (NVL)", "batch_size": 64, "throughput_tok_s": 65, "cost_per_1m_tokens": 1.85, "vram_headroom_pct": 47, "fit_score": 90, "notes": "1.5TB VRAM, excellent fit for mega-models"},
            "MI300X": {"optimal_config": "8x MI300X", "batch_size": 32, "throughput_tok_s": 35, "cost_per_1m_tokens": 2.20, "vram_headroom_pct": 47, "fit_score": 78, "notes": "1.5TB HBM, competitive AMD option"},
            "GB200": {"optimal_config": "4x GB200", "batch_size": 128, "throughput_tok_s": 95, "cost_per_1m_tokens": 1.40, "vram_headroom_pct": 47, "fit_score": 95, "notes": "NVL72 rack-scale, best for 400B+ models"},
            "H200": {"optimal_config": "8x H200", "batch_size": 16, "throughput_tok_s": 32, "cost_per_1m_tokens": 2.50, "vram_headroom_pct": 28, "fit_score": 65, "notes": "1.13TB total, feasible but limited headroom"}
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
