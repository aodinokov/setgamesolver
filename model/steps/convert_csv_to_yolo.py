#!/usr/bin/env python3
"""
Convert CSV annotation format (from TensorFlow Object Detection API) to YOLO format.

Input CSV format:
    filename,width,height,class,xmin,ymin,xmax,ymax
    image.jpg,800,600,setgame-card,96,117,217,201

Output YOLO format:
    wrk_d_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Label files contain: class_id x_center y_center width height (normalized 0-1)
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


class CSVToYOLOConverter:
    """Convert CSV annotations to YOLO format"""
    
    def __init__(self, output_base_dir="wrk_d"):
        self.output_base_dir = output_base_dir
        self.images_base_dir = os.path.join(output_base_dir, "images")
        self.labels_base_dir = os.path.join(output_base_dir, "labels")
        
        # Class mapping
        self.class_map = {"setgame-card": 0}
    
    def create_directories(self):
        """Create YOLO directory structure"""
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.images_base_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_base_dir, split), exist_ok=True)
        print("✓ Created YOLO directory structure")
    
    def normalize_bbox(self, bbox, width, height):
        """
        Convert bounding box from (xmin, ymin, xmax, ymax) to (x_center, y_center, width, height)
        Both in pixel coordinates, output normalized to 0-1
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Convert to center format
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # Normalize to 0-1
        x_center_norm = x_center / width
        y_center_norm = y_center / height
        width_norm = bbox_width / width
        height_norm = bbox_height / height
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def read_csv(self, csv_path):
        """Read CSV and group annotations by image"""
        annotations = defaultdict(list)
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                width = int(row['width'])
                height = int(row['height'])
                class_name = row['class']
                bbox = (
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax'])
                )
                
                annotations[filename].append({
                    'width': width,
                    'height': height,
                    'class': class_name,
                    'bbox': bbox
                })
        
        return annotations
    
    def convert(self, train_csv, test_csv, images_dir, copy_images=False):
        """
        Convert CSV annotations to YOLO format
        
        Args:
            train_csv: Path to training CSV file
            test_csv: Path to test CSV file
            images_dir: Directory containing images
            copy_images: If True, copy images to YOLO structure; if False, assume they're already there
        """
        print("\n=== Converting CSV to YOLO Format ===\n")
        
        self.create_directories()
        
        # Read annotations
        print("Reading annotations from CSV files...")
        train_annotations = self.read_csv(train_csv)
        test_annotations = self.read_csv(test_csv)
        
        print(f"  Train: {len(train_annotations)} images")
        print(f"  Test:  {len(test_annotations)} images")
        
        # Convert train set
        print("\nConverting train set...")
        self._convert_split(train_annotations, images_dir, "train", copy_images)
        
        # Convert test set (use as validation)
        print("\nConverting test set (using as validation)...")
        self._convert_split(test_annotations, images_dir, "val", copy_images)
        
        print("\n✓ Conversion completed!")
        print(f"\nOutput structure:")
        print(f"  {self.images_base_dir}/train/  <- training images")
        print(f"  {self.images_base_dir}/val/    <- validation images")
        print(f"  {self.labels_base_dir}/train/  <- training labels")
        print(f"  {self.labels_base_dir}/val/    <- validation labels")
    
    def _convert_split(self, annotations, images_dir, split, copy_images):
        """Convert a single split (train/val)"""
        labels_dir = os.path.join(self.labels_base_dir, split)
        images_split_dir = os.path.join(self.images_base_dir, split)
        
        count = 0
        errors = []
        
        for filename, image_annotations in annotations.items():
            src_image_path = os.path.join(images_dir, filename)
            
            # Check if image exists
            if not os.path.exists(src_image_path):
                errors.append(f"Image not found: {src_image_path}")
                continue
            
            # Copy or link image if requested
            if copy_images:
                dst_image_path = os.path.join(images_split_dir, filename)
                if not os.path.exists(dst_image_path):
                    shutil.copy2(src_image_path, dst_image_path)
            
            # Get image dimensions from first annotation (they're all the same for an image)
            width = image_annotations[0]['width']
            height = image_annotations[0]['height']
            
            # Create label file
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for annotation in image_annotations:
                    bbox = annotation['bbox']
                    class_name = annotation['class']
                    
                    # Get class ID
                    if class_name not in self.class_map:
                        errors.append(f"Unknown class '{class_name}' in {filename}")
                        continue
                    
                    class_id = self.class_map[class_name]
                    
                    # Normalize bounding box
                    x_center, y_center, bbox_width, bbox_height = self.normalize_bbox(
                        bbox, width, height
                    )
                    
                    # Write YOLO format
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            count += 1
        
        print(f"  ✓ Converted {count} images")
        
        if errors:
            print(f"  ⚠ {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        
        return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV annotations (TF Object Detection format) to YOLO format"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="wrk_d/train_labels.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="wrk_d/test_labels.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="wrk_d/images",
        help="Directory containing image files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wrk_d_yolo",
        help="Output directory for YOLO structure (will create images/ and labels/ subdirs)"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images to YOLO structure (default: just reference existing images)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.train_csv):
        print(f"Error: Training CSV not found: {args.train_csv}")
        return 1
    
    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV not found: {args.test_csv}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    # Convert
    converter = CSVToYOLOConverter(output_base_dir=args.output_dir)
    converter.convert(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        images_dir=args.images_dir,
        copy_images=args.copy_images
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
