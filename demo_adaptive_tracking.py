#!/usr/bin/env python3
"""
Adaptive Fluid Surface Tracking Demo
====================================

This script demonstrates the adaptive fluid tracking system that works
across different videos with varying lighting conditions and fluid colors.

Features:
- Interactive color selection by clicking on the fluid
- Automatic color profile saving/loading
- Adaptive lighting adjustment
- Real-time parameter tuning
- Comprehensive analysis plots

Usage:
    python demo_adaptive_tracking.py
"""

import os
import sys
from fluid_tracking import FluidTracker

def demo_basic_usage():
    """Basic usage demonstration"""
    print("=== BASIC USAGE DEMO ===")
    print("This demo shows the basic workflow:")
    print("1. Load video")
    print("2. Select ROI")
    print("3. Select fluid color")
    print("4. Analyze motion")
    print("5. View results")
    
    # Example with a specific video
    video_path = "imapttiG_1_prima parte.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found. Please check the file path.")
        return
    
    # Create tracker
    tracker = FluidTracker(video_path, f"demo_output_{os.path.splitext(video_path)[0]}.mp4")
    
    # Ask user for mode
    print("\nChoose analysis mode:")
    print("1. Full Frame Mode (no ROI selection)")
    print("2. ROI Mode (manual selection)")
    
    mode_choice = input("Enter mode (1 or 2): ").strip()
    use_full_frame = mode_choice == "1"
    
    # Run analysis
    tracker.analyze_wave_motion(debug_mode=True, use_full_frame=use_full_frame)
    
    # Show plots
    tracker.plot_surface_analysis()

def demo_multiple_videos():
    """Demo for handling multiple videos with different conditions"""
    print("\n=== MULTIPLE VIDEOS DEMO ===")
    print("This demo shows how to process multiple videos with different lighting:")
    
    # List of videos to process
    test_videos = [
        "imapttiG_1_prima parte.mp4",
        "MicroG_Impatto_4.mp4",
        "colorante.mp4"
    ]
    
    for video in test_videos:
        if os.path.exists(video):
            print(f"\nProcessing: {video}")
            print("=" * 50)
            
            # Create tracker for this video
            tracker = FluidTracker(video, f"demo_output_{os.path.splitext(video)[0]}.mp4")
            
            # Use full frame mode for batch processing
            print("Using Full Frame Mode for batch processing")
            
            # Run analysis (color profiles will be saved/loaded automatically)
            tracker.analyze_wave_motion(debug_mode=False, use_full_frame=True)
            
            # Show analysis
            tracker.plot_surface_analysis()
            
            print(f"Completed: {video}")
        else:
            print(f"Skipping {video} - file not found")

def demo_color_profile_management():
    """Demo for color profile management"""
    print("\n=== COLOR PROFILE MANAGEMENT DEMO ===")
    print("This demo shows how color profiles are automatically saved and loaded:")
    
    video_path = "imapttiG_1_prima parte.mp4"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return
    
    # Create tracker
    tracker = FluidTracker(video_path)
    
    # Check if color profile exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    profile_path = os.path.join(tracker.color_profiles_dir, f"{video_name}_color_profile.json")
    
    if os.path.exists(profile_path):
        print(f"Color profile exists: {profile_path}")
        
        # Load and display profile info
        if tracker.load_color_profile(video_name):
            print(f"Target HSV color: {tracker.target_color_hsv}")
            print(f"HSV ranges: {tracker.adaptive_hsv_ranges}")
    else:
        print("No color profile found. You'll need to select colors manually.")
    
    print("\nDuring analysis, you can:")
    print("- Press 'c' to reselect colors")
    print("- Press 'd' to toggle debug mode")
    print("- Color profiles are automatically saved after selection")

def demo_advanced_features():
    """Demo of advanced features"""
    print("\n=== ADVANCED FEATURES DEMO ===")
    print("Advanced features include:")
    print("1. Adaptive lighting adjustment")
    print("2. Hue wrap-around handling")
    print("3. Container edge filtering")
    print("4. Surface interpolation")
    print("5. Motion vector tracking")
    
    # Show color profiles directory
    if os.path.exists("color_profiles"):
        print(f"\nColor profiles directory: color_profiles/")
        profiles = [f for f in os.listdir("color_profiles") if f.endswith(".json")]
        if profiles:
            print("Existing profiles:")
            for profile in profiles:
                print(f"  - {profile}")
        else:
            print("  No profiles found")

def interactive_demo():
    """Interactive demo with user choices"""
    print("\n=== INTERACTIVE DEMO ===")
    print("Choose what you want to demo:")
    print("1. Basic usage (single video)")
    print("2. Multiple videos")
    print("3. Color profile management")
    print("4. Advanced features info")
    print("5. Process custom video")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            demo_basic_usage()
        elif choice == "2":
            demo_multiple_videos()
        elif choice == "3":
            demo_color_profile_management()
        elif choice == "4":
            demo_advanced_features()
        elif choice == "5":
            custom_video = input("Enter video path: ").strip()
            if os.path.exists(custom_video):
                tracker = FluidTracker(custom_video)
                
                # Ask for mode
                print("\nChoose analysis mode:")
                print("1. Full Frame Mode (no ROI selection)")
                print("2. ROI Mode (manual selection)")
                
                mode_choice = input("Enter mode (1 or 2): ").strip()
                use_full_frame = mode_choice == "1"
                
                tracker.analyze_wave_motion(debug_mode=True, use_full_frame=use_full_frame)
                tracker.plot_surface_analysis()
            else:
                print("Video not found!")
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

def main():
    """Main demo function"""
    print("🌊 ADAPTIVE FLUID SURFACE TRACKING DEMO 🌊")
    print("=" * 60)
    print("This system automatically adapts to different:")
    print("✓ Lighting conditions")
    print("✓ Fluid colors")
    print("✓ Video qualities")
    print("✓ Container shapes")
    print("=" * 60)
    
    # Check if we have any videos
    sample_videos = [
        "imapttiG_1_prima parte.mp4",
        "imapttiG_1_seconda_parte.mp4",
        "MicroG_Impatto_4.mp4",
        "colorante.mp4",
        "exported.mp4"
    ]
    
    available_videos = [v for v in sample_videos if os.path.exists(v)]
    
    if not available_videos:
        print("⚠️  No sample videos found in current directory.")
        print("Please add some video files to test the system.")
        return
    
    print(f"Found {len(available_videos)} videos to work with:")
    for video in available_videos:
        print(f"  ✓ {video}")
    
    # Run interactive demo
    interactive_demo()

if __name__ == "__main__":
    main() 