#!/usr/bin/env python3
"""
Generate synthetic scanned-style Indian GST invoice image prompts.

This script composes prompts from reusable fragments and optionally generates
images via Azure OpenAI DALL-E compatible image endpoint.
"""

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

try:
    from dotenv import load_dotenv
    # Load nearest .env (current working dir or project root)
    load_dotenv()
except Exception:
    pass  # Safe to ignore if not installed yet

class ImagePromptGenerator:
    """Generator for invoice image prompts and optional images."""
    
    def __init__(self, fragments_file: Path, model: str = "gpt-image-1",
                 size: str = "1024x1792", timeout: int = 60,
                 deployment: Optional[str] = None, num_images: int = 1):
        """Initialize the generator with configuration."""
        self.fragments_file = fragments_file
        self.model = model
        self.size = size
        self.timeout = timeout
        self.fragments = self._load_fragments()

        # Azure OpenAI DALL-E specific configuration (dedicated env vars)
        self.endpoint = os.getenv("AZURE_OPENAI_DALLE_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_DALLE_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DALLE_DEPLOYMENT", "dalle")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.num_images = max(1, min(num_images, 10))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_fragments(self) -> Dict[str, List[str]]:
        """Load prompt fragments from YAML file."""
        try:
            with open(self.fragments_file, 'r', encoding='utf-8') as f:
                fragments = yaml.safe_load(f)
            
            # Ensure all required sections exist
            required_sections = ['base', 'layout', 'tax', 'noise', 'embellishments', 'variants']
            for section in required_sections:
                if section not in fragments:
                    self.logger.warning(f"Missing section '{section}' in fragments file")
                    fragments[section] = []
                    
            return fragments
            
        except FileNotFoundError:
            self.logger.error(f"Fragments file not found: {self.fragments_file}")
            return {section: [] for section in ['base', 'layout', 'tax', 'noise', 'embellishments', 'variants']}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file: {e}")
            return {section: [] for section in ['base', 'layout', 'tax', 'noise', 'embellishments', 'variants']}
    
    def _compose_prompt(self, used_prompts: set) -> Tuple[str, Dict[str, str]]:
        """Compose a unique prompt from fragments."""
        max_attempts = 50  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            # Select fragments according to rules:
            # 1 base + 1 layout + 1 tax + 0-1 noise + 0-1 embellishment + 0-2 variants
            selected_fragments = {}
            
            # Required: 1 from each of base, layout, tax
            if self.fragments['base']:
                selected_fragments['base'] = random.choice(self.fragments['base'])
            
            if self.fragments['layout']:
                selected_fragments['layout'] = random.choice(self.fragments['layout'])
                
            if self.fragments['tax']:
                selected_fragments['tax'] = random.choice(self.fragments['tax'])
            
            # Optional: 0-1 noise (50% chance)
            if self.fragments['noise'] and random.choice([True, False]):
                selected_fragments['noise'] = random.choice(self.fragments['noise'])
            
            # Optional: 0-1 embellishment (40% chance)
            if self.fragments['embellishments'] and random.random() < 0.4:
                selected_fragments['embellishments'] = random.choice(self.fragments['embellishments'])
            
            # Optional: 0-2 variants (60% chance for first, 30% for second)
            variants = []
            if self.fragments['variants']:
                if random.random() < 0.6:
                    variants.append(random.choice(self.fragments['variants']))
                if random.random() < 0.3:
                    # Ensure we don't pick the same variant twice
                    remaining_variants = [v for v in self.fragments['variants'] if v not in variants]
                    if remaining_variants:
                        variants.append(random.choice(remaining_variants))
            
            # Compose the final prompt
            prompt_parts = []
            for key in ['base', 'layout', 'tax', 'noise', 'embellishments']:
                if key in selected_fragments:
                    prompt_parts.append(selected_fragments[key])
            
            # Add variants
            prompt_parts.extend(variants)
            
            # Join with periods and add fixed suffix
            prompt = '. '.join(prompt_parts)
            if prompt and not prompt.endswith('.'):
                prompt += '.'
            
            # Add fixed suffix
            suffix = " All data must be obviously fictional. Use placeholder logo. Add footer 'Synthetic Test Data'."
            prompt += suffix
            
            # Clean up punctuation (avoid double periods)
            prompt = re.sub(r'\.+', '.', prompt)
            prompt = re.sub(r'\s+', ' ', prompt).strip()
            
            # Check for uniqueness
            if prompt not in used_prompts:
                # Track which fragments were selected
                fragment_info = dict(selected_fragments)
                if variants:
                    fragment_info['variants'] = variants
                return prompt, fragment_info
        
        # If we couldn't find a unique prompt, return the last one anyway
        self.logger.warning("Could not generate unique prompt after maximum attempts")
        return prompt, fragment_info
    
    def _call_image_api(self, prompt: str) -> Tuple[Optional[List[bytes]], Dict, float]:
        """Call Azure OpenAI Image Generation API."""
        if not self.endpoint or not self.api_key or not self.deployment:
            return None, {"error": "Missing Azure DALL-E credentials (AZURE_OPENAI_DALLE_*)"}, 0.0

        # url = f"{self.endpoint}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "model": self.deployment,
            "prompt": prompt,
            "size": self.size,
            "n": self.num_images,
            "response_format": "b64_json"
        }
        
        start_time = time.time()
        
        # Retry logic with exponential backoff
        max_retries = 2
        for retry in range(max_retries + 1):
            try:
                # Print the request URL and parameters for debugging
                print(f"Requesting: {self.endpoint}?api-version={self.api_version}")
                print(f"Headers: {headers}")
                print(f"Payload: {json.dumps(payload)}")

                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if response.status_code in (200, 201):
                    data = response.json()
                    # Some Azure deployments return direct data; others require follow-up (omitted for brevity)
                    if "data" in data and data["data"]:
                        images: List[bytes] = []
                        for item in data["data"]:
                            if "b64_json" in item:
                                try:
                                    images.append(base64.b64decode(item["b64_json"]))
                                except Exception:
                                    continue
                        if images:
                            return images, {"status": "success", "count": len(images), "response": data}, latency
                        return None, {"error": "No decodable image data", "response": data}, latency
                    return None, {"error": "Malformed response", "response": data}, latency
                
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # Transient errors - retry with exponential backoff
                    if retry < max_retries:
                        wait_time = (2 ** retry) * 1.0  # 1, 2, 4 seconds
                        self.logger.warning(f"API error {response.status_code}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                # Non-retryable error or max retries exceeded
                return None, {
                    "error": f"API error {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }, latency
                
            except requests.exceptions.Timeout:
                if retry < max_retries:
                    wait_time = (2 ** retry) * 1.0
                    self.logger.warning(f"Request timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None, {"error": "Request timeout"}, (time.time() - start_time) * 1000
                
            except requests.exceptions.RequestException as e:
                return None, {"error": f"Request failed: {str(e)}"}, (time.time() - start_time) * 1000
        
        return None, {"error": "Max retries exceeded"}, (time.time() - start_time) * 1000
    
    def generate_samples(self, count: int, output_dir: Path, generate_images: bool = False,
                        dry_run: bool = False, no_metadata: bool = False, seed: Optional[int] = None) -> None:
        """Generate prompt samples and optionally images."""
        
        if seed is not None:
            random.seed(seed)
        
        # Check Azure credentials if image generation requested
        if generate_images and (not self.endpoint or not self.api_key):
            self.logger.warning("Missing Azure OpenAI credentials - only generating prompts")
            generate_images = False
        
        # Create output directories
        if not dry_run:
            (output_dir / "prompts").mkdir(parents=True, exist_ok=True)
            if generate_images:
                (output_dir / "images").mkdir(parents=True, exist_ok=True)
            if not no_metadata:
                (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        used_prompts = set()
        
        self.logger.info(f"Generating {count} samples...")
        
        for i in range(1, count + 1):
            self.logger.info(f"Processing sample {i}/{count}")
            
            # Compose prompt
            prompt, fragments = self._compose_prompt(used_prompts)
            used_prompts.add(prompt)
            
            if dry_run:
                print(f"\n--- Sample {i} ---")
                print(f"Prompt: {prompt}")
                print(f"Fragments: {fragments}")
                continue
            
            # Save prompt file
            prompt_file = output_dir / "prompts" / f"invoice_prompt_{i:03d}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Generate image if requested
            image_files: List[str] = []
            api_response = {}
            api_latency = 0.0
            
            if generate_images:
                self.logger.info(f"Generating image for sample {i}")
                images_bytes, api_response, api_latency = self._call_image_api(prompt)
                if images_bytes:
                    for idx, img in enumerate(images_bytes, start=1):
                        image_path = output_dir / "images" / f"invoice_image_{i:03d}_{idx}.png"
                        with open(image_path, 'wb') as f:
                            f.write(img)
                        image_files.append(image_path.name)
                    self.logger.info(f"Saved {len(image_files)} image(s) for sample {i}")
                else:
                    self.logger.error(f"Failed image gen sample {i}: {api_response.get('error', 'Unknown error')}")
            
            # Save metadata if not disabled
            if not no_metadata:
                metadata = {
                    "prompt_file": str(prompt_file.name),
                    "image_files": image_files if image_files else None,
                    "fragments": fragments,
                    "seed": seed,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    "deployment": self.deployment,
                    "model": self.model,
                    "size": self.size,
                    "num_images_requested": self.num_images,
                    "api_latency_ms": round(api_latency, 2),
                    "status": "success" if (not generate_images or image_files) else "failed"
                }
                
                # Add minimal API response info if available
                if api_response:
                    metadata["api_response"] = {
                        "status": api_response.get("status", "error"),
                        "error": api_response.get("error")
                    }
                
                metadata_file = output_dir / "metadata" / f"invoice_metadata_{i:03d}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generation complete! Files saved to: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian GST invoice image prompts and optionally images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 8 --generate-images --seed 42
  %(prog)s --count 5 --dry-run
  %(prog)s --count 3 --size 832x1216
  %(prog)s --count 10 --fragments custom_fragments.yml --output-dir /custom/path
        """
    )
    
    parser.add_argument('--count', type=int, default=5,
                       help='Number of samples to generate (default: 5)')
    parser.add_argument('--fragments', type=Path, 
                       default=Path('datasets/indian_gst/prompt_fragments.yml'),
                       help='Path to YAML fragments file (default: datasets/indian_gst/prompt_fragments.yml)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('datasets/indian_gst/image_invoices'),
                       help='Output directory (default: datasets/indian_gst/image_invoices)')
    parser.add_argument('--model', default='gpt-image-1',
                       help='(Logical) model label (default: gpt-image-1)')
    parser.add_argument('--deployment', default=None,
                       help='Azure DALL-E deployment name (default: env AZURE_OPENAI_DALLE_DEPLOYMENT or "dalle")')
    parser.add_argument('--size', default='1024x1792',
                       help='Image size (default: 1024x1792)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for deterministic fragment selection')
    parser.add_argument('--generate-images', action='store_true',
                       help='Generate images via Azure OpenAI DALL-E API (requires AZURE_OPENAI_DALLE_* env vars)')
    parser.add_argument('--num-images', type=int, default=1,
                       help='Number of images per prompt (1-10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show composed prompts to stdout only')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip generating metadata JSON files')
    parser.add_argument('--timeout', type=int, default=60,
                       help='API timeout in seconds (default: 60)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.count <= 0:
        logger.error("Count must be positive")
        sys.exit(1)
    
    if not args.fragments.exists():
        logger.error(f"Fragments file not found: {args.fragments}")
        sys.exit(1)
    
    # Validate size format
    if not re.match(r'^\d+x\d+$', args.size):
        logger.error("Size must be in format WIDTHxHEIGHT (e.g., 1024x1536)")
        sys.exit(1)
    
    # Show environment info
    if args.generate_images:
        endpoint = os.getenv("AZURE_OPENAI_DALLE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_DALLE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DALLE_DEPLOYMENT") or args.deployment or "dalle"
        
        if not endpoint or not api_key:
            logger.warning("Missing Azure OpenAI environment variables:")
            logger.warning("  AZURE_OPENAI_ENDPOINT")
            logger.warning("  AZURE_OPENAI_API_KEY")
            logger.warning("Will generate prompts only")
        else:
            logger.info(f"Azure DALL-E endpoint: {endpoint}")
            logger.info(f"Deployment: {deployment}")
            logger.info(f"API version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')}")
    
    try:
        generator = ImagePromptGenerator(
            fragments_file=args.fragments,
            model=args.model,
            size=args.size,
            timeout=args.timeout,
            deployment=args.deployment,
            num_images=args.num_images
        )
        
        generator.generate_samples(
            count=args.count,
            output_dir=args.output_dir,
            generate_images=args.generate_images,
            dry_run=args.dry_run,
            no_metadata=args.no_metadata,
            seed=args.seed
        )
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()