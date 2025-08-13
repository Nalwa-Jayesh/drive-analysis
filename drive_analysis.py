import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import yt_dlp
from pathlib import Path


@dataclass
class PoseMetrics:
    """Container for pose estimation metrics"""

    front_elbow_angle: Optional[float] = None
    spine_lean: Optional[float] = None
    head_knee_alignment: Optional[float] = None
    front_foot_angle: Optional[float] = None
    frame_number: int = 0
    timestamp: float = 0.0


class CricketPoseAnalyzer:
    """Main class for cricket pose analysis"""

    def __init__(self, config: Dict = None):
        """Initialize the analyzer with configuration"""
        self.config = config or self._default_config()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Metrics tracking
        self.metrics_history: List[PoseMetrics] = []
        self.phase_history: List[str] = []
        self.current_phase = "Stance"

        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.total_frames = 0

    def _default_config(self) -> Dict:
        """Default configuration for analysis thresholds"""
        return {
            "thresholds": {
                "good_elbow_angle": (100, 140),
                "good_spine_lean": (5, 25),
                "good_alignment_threshold": 50,
                "good_foot_angle": (-30, 30),
            },
            "output_dir": "output",
            "video_fps": 30,
            "phases": [
                "Stance",
                "Stride",
                "Downswing",
                "Impact",
                "Follow-through",
                "Recovery",
            ],
        }

    def download_video(self, url: str, output_path: str = "input_video.mp4") -> str:
        """Download video from YouTube URL"""
        print(f"Downloading video from: {url}")

        ydl_opts = {
            "format": "best[height<=720]",  # Limit quality for faster processing
            "outtmpl": output_path,
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Video downloaded successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p1-p2-p3)"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = np.arccos(cos_angle) * 180 / np.pi

            return angle
        except:
            return None

    def calculate_spine_lean(self, hip: np.ndarray, shoulder: np.ndarray) -> float:
        """Calculate spine lean angle from vertical"""
        try:
            spine_vector = shoulder - hip
            vertical_vector = np.array([0, -1])  # Pointing up

            cos_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi

            return angle
        except:
            return None

    def calculate_head_knee_alignment(
        self, head: np.ndarray, knee: np.ndarray
    ) -> float:
        """Calculate horizontal distance between head and front knee"""
        try:
            return abs(head[0] - knee[0])  # Horizontal distance in pixels
        except:
            return None

    def extract_pose_metrics(
        self, landmarks, frame_shape: Tuple[int, int]
    ) -> PoseMetrics:
        """Extract biomechanical metrics from pose landmarks"""
        height, width = frame_shape[:2]
        metrics = PoseMetrics()

        if not landmarks:
            return metrics

        # Convert normalized coordinates to pixel coordinates
        def get_landmark_coords(landmark_idx: int) -> Optional[np.ndarray]:
            if landmark_idx < len(landmarks.landmark):
                lm = landmarks.landmark[landmark_idx]
                if lm.visibility > 0.5:  # Only use visible landmarks
                    return np.array([lm.x * width, lm.y * height])
            return None

        # Key landmark indices for MediaPipe Pose
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        NOSE = 0

        # Get landmark coordinates
        left_shoulder = get_landmark_coords(LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(RIGHT_SHOULDER)
        left_elbow = get_landmark_coords(LEFT_ELBOW)
        right_elbow = get_landmark_coords(RIGHT_ELBOW)
        left_wrist = get_landmark_coords(LEFT_WRIST)
        right_wrist = get_landmark_coords(RIGHT_WRIST)
        left_hip = get_landmark_coords(LEFT_HIP)
        right_hip = get_landmark_coords(RIGHT_HIP)
        left_knee = get_landmark_coords(LEFT_KNEE)
        right_knee = get_landmark_coords(RIGHT_KNEE)
        left_ankle = get_landmark_coords(LEFT_ANKLE)
        right_ankle = get_landmark_coords(RIGHT_ANKLE)
        nose = get_landmark_coords(NOSE)

        # Determine front side (assume left side is front for right-handed batsman)
        # In a real implementation, this could be auto-detected
        front_shoulder = left_shoulder
        front_elbow = left_elbow
        front_wrist = left_wrist
        front_knee = left_knee
        front_ankle = left_ankle

        # Calculate front elbow angle
        if all(
            [
                front_shoulder is not None,
                front_elbow is not None,
                front_wrist is not None,
            ]
        ):
            metrics.front_elbow_angle = self.calculate_angle(
                front_shoulder, front_elbow, front_wrist
            )

        # Calculate spine lean
        if (
            left_hip is not None
            and right_hip is not None
            and left_shoulder is not None
            and right_shoulder is not None
        ):
            hip_center = (left_hip + right_hip) / 2
            shoulder_center = (left_shoulder + right_shoulder) / 2
            metrics.spine_lean = self.calculate_spine_lean(hip_center, shoulder_center)

        # Calculate head-knee alignment
        if nose is not None and front_knee is not None:
            metrics.head_knee_alignment = self.calculate_head_knee_alignment(
                nose, front_knee
            )

        # Calculate front foot angle (simplified - angle of foot relative to horizontal)
        if front_knee is not None and front_ankle is not None:
            foot_vector = front_ankle - front_knee
            horizontal_vector = np.array([1, 0])
            cos_angle = np.dot(foot_vector, horizontal_vector) / (
                np.linalg.norm(foot_vector) * np.linalg.norm(horizontal_vector)
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            metrics.front_foot_angle = np.arccos(cos_angle) * 180 / np.pi

        return metrics

    def detect_phase(self, metrics: PoseMetrics, frame_idx: int) -> str:
        """Simple phase detection based on metrics (bonus feature)"""
        # This is a simplified heuristic - in reality, you'd use more sophisticated ML
        if not metrics.front_elbow_angle:
            return "Stance"

        elbow_angle = metrics.front_elbow_angle

        if elbow_angle > 150:
            return "Stance"
        elif elbow_angle > 120:
            return "Stride"
        elif elbow_angle > 90:
            return "Downswing"
        elif elbow_angle > 70:
            return "Impact"
        elif elbow_angle > 60:
            return "Follow-through"
        else:
            return "Recovery"

    def draw_overlays(
        self, frame: np.ndarray, landmarks, metrics: PoseMetrics, phase: str, fps: float
    ) -> np.ndarray:
        """Draw pose skeleton and metric overlays on frame"""
        overlay_frame = frame.copy()

        # Draw pose skeleton
        if landmarks:
            self.mp_drawing.draw_landmarks(
                overlay_frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw metrics overlay
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2

        # Phase indicator
        cv2.putText(
            overlay_frame,
            f"Phase: {phase}",
            (10, y_offset),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )
        y_offset += 30

        # FPS indicator
        cv2.putText(
            overlay_frame,
            f"FPS: {fps:.1f}",
            (10, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        y_offset += 30

        # Elbow angle
        if metrics.front_elbow_angle is not None:
            angle_text = f"Elbow: {metrics.front_elbow_angle:.1f}°"
            if self._is_good_elbow_angle(metrics.front_elbow_angle):
                angle_text += " ✅"
                color = (0, 255, 0)  # Green
            else:
                angle_text += " ❌"
                color = (0, 0, 255)  # Red

            cv2.putText(
                overlay_frame,
                angle_text,
                (10, y_offset),
                font,
                font_scale,
                color,
                thickness,
            )
            y_offset += 30

        # Spine lean
        if metrics.spine_lean is not None:
            lean_text = f"Spine: {metrics.spine_lean:.1f}°"
            if self._is_good_spine_lean(metrics.spine_lean):
                lean_text += " ✅"
                color = (0, 255, 0)
            else:
                lean_text += " ❌"
                color = (0, 0, 255)

            cv2.putText(
                overlay_frame,
                lean_text,
                (10, y_offset),
                font,
                font_scale,
                color,
                thickness,
            )
            y_offset += 30

        # Head-knee alignment
        if metrics.head_knee_alignment is not None:
            alignment_text = f"Alignment: {metrics.head_knee_alignment:.1f}px"
            if self._is_good_alignment(metrics.head_knee_alignment):
                alignment_text += " ✅"
                color = (0, 255, 0)
            else:
                alignment_text += " ❌"
                color = (0, 0, 255)

            cv2.putText(
                overlay_frame,
                alignment_text,
                (10, y_offset),
                font,
                font_scale,
                color,
                thickness,
            )
            y_offset += 30

        return overlay_frame

    def _is_good_elbow_angle(self, angle: float) -> bool:
        """Check if elbow angle is within good range"""
        return (
            self.config["thresholds"]["good_elbow_angle"][0]
            <= angle
            <= self.config["thresholds"]["good_elbow_angle"][1]
        )

    def _is_good_spine_lean(self, angle: float) -> bool:
        """Check if spine lean is within good range"""
        return (
            self.config["thresholds"]["good_spine_lean"][0]
            <= angle
            <= self.config["thresholds"]["good_spine_lean"][1]
        )

    def _is_good_alignment(self, distance: float) -> bool:
        """Check if head-knee alignment is good"""
        return distance <= self.config["thresholds"]["good_alignment_threshold"]

    def process_video(self, video_path: str) -> Dict:
        """Process entire video and generate annotated output"""
        print(f"Processing video: {video_path}")

        # Setup video capture and writer
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video specs: {width}x{height}, {fps} FPS, {frame_count} frames")

        # Setup output directory
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(exist_ok=True)

        # Setup video writer
        output_path = output_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        start_time = time.time()
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start = time.time()

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform pose estimation
                results = self.pose.process(rgb_frame)

                # Extract metrics
                metrics = self.extract_pose_metrics(results.pose_landmarks, frame.shape)
                metrics.frame_number = frame_idx
                metrics.timestamp = frame_idx / fps

                # Detect phase
                phase = self.detect_phase(metrics, frame_idx)

                # Store metrics
                self.metrics_history.append(metrics)
                self.phase_history.append(phase)

                # Calculate current FPS
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                current_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0

                # Draw overlays
                annotated_frame = self.draw_overlays(
                    frame, results.pose_landmarks, metrics, phase, current_fps
                )

                # Write frame
                out.write(annotated_frame)

                frame_idx += 1

                # Progress update
                if frame_idx % 30 == 0:
                    progress = (frame_idx / frame_count) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count})")

        finally:
            cap.release()
            out.release()

        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time

        print(f"Processing complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_path}")

        # Generate evaluation
        evaluation = self.generate_evaluation()

        # Save evaluation
        eval_path = output_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        print(f"Evaluation saved to: {eval_path}")

        # Generate temporal chart (bonus feature)
        self.generate_temporal_chart(output_dir)

        return {
            "output_video": str(output_path),
            "evaluation_file": str(eval_path),
            "processing_stats": {
                "total_frames": frame_idx,
                "total_time": total_time,
                "average_fps": avg_fps,
                "target_fps_achieved": avg_fps >= 10,
            },
            "evaluation": evaluation,
        }

    def analyze_consistency_issues(self) -> Dict:
        """Analyze and diagnose consistency issues"""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        # Extract valid metrics
        elbow_angles = [
            m.front_elbow_angle
            for m in self.metrics_history
            if m.front_elbow_angle is not None
        ]
        spine_leans = [
            m.spine_lean for m in self.metrics_history if m.spine_lean is not None
        ]
        alignments = [
            m.head_knee_alignment
            for m in self.metrics_history
            if m.head_knee_alignment is not None
        ]

        analysis = {
            "total_frames": len(self.metrics_history),
            "valid_elbow_detections": len(elbow_angles),
            "valid_spine_detections": len(spine_leans),
            "valid_alignment_detections": len(alignments),
            "detection_rates": {
                "elbow": len(elbow_angles) / len(self.metrics_history) * 100
                if self.metrics_history
                else 0,
                "spine": len(spine_leans) / len(self.metrics_history) * 100
                if self.metrics_history
                else 0,
                "alignment": len(alignments) / len(self.metrics_history) * 100
                if self.metrics_history
                else 0,
            },
        }

        if elbow_angles:
            analysis["elbow_stats"] = {
                "mean": np.mean(elbow_angles),
                "std": np.std(elbow_angles),
                "min": np.min(elbow_angles),
                "max": np.max(elbow_angles),
                "range": np.max(elbow_angles) - np.min(elbow_angles),
            }

        if spine_leans:
            analysis["spine_stats"] = {
                "mean": np.mean(spine_leans),
                "std": np.std(spine_leans),
                "min": np.min(spine_leans),
                "max": np.max(spine_leans),
                "range": np.max(spine_leans) - np.min(spine_leans),
            }

        if alignments:
            analysis["alignment_stats"] = {
                "mean": np.mean(alignments),
                "std": np.std(alignments),
                "min": np.min(alignments),
                "max": np.max(alignments),
                "range": np.max(alignments) - np.min(alignments),
            }

        return analysis

    def generate_evaluation(self) -> Dict:
        """Generate final shot evaluation with scores and feedback"""
        if not self.metrics_history:
            return {"error": "No metrics available for evaluation"}

        # Extract valid metrics
        elbow_angles = [
            m.front_elbow_angle
            for m in self.metrics_history
            if m.front_elbow_angle is not None
        ]
        spine_leans = [
            m.spine_lean for m in self.metrics_history if m.spine_lean is not None
        ]
        alignments = [
            m.head_knee_alignment
            for m in self.metrics_history
            if m.head_knee_alignment is not None
        ]
        foot_angles = [
            m.front_foot_angle
            for m in self.metrics_history
            if m.front_foot_angle is not None
        ]

        evaluation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames_analyzed": len(self.metrics_history),
            "scores": {},
            "feedback": {},
            "phase_distribution": {},
            "summary_stats": {},
        }

        # Phase distribution
        phase_counts = {}
        for phase in self.phase_history:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        total_phases = len(self.phase_history)
        evaluation["phase_distribution"] = {
            phase: (count / total_phases) * 100 for phase, count in phase_counts.items()
        }

        # Footwork score (based on phase transitions and stability)
        footwork_score = 7.0  # Base score
        if len(set(self.phase_history)) >= 4:  # Good phase progression
            footwork_score += 1.5
        if alignments:
            avg_alignment = np.mean(alignments)
            if avg_alignment < 30:  # Good alignment
                footwork_score += 1.5

        evaluation["scores"]["footwork"] = min(10, max(1, footwork_score))
        evaluation["feedback"]["footwork"] = [
            "Good phase progression through the shot"
            if len(set(self.phase_history)) >= 4
            else "Work on smoother transitions between phases",
            "Maintain balance on front foot"
            if alignments and np.mean(alignments) > 40
            else "Good foot positioning",
        ]

        # Head Position score
        head_score = 7.0
        if alignments:
            good_alignment_ratio = sum(1 for a in alignments if a <= 50) / len(
                alignments
            )
            head_score += good_alignment_ratio * 3

        evaluation["scores"]["head_position"] = min(10, max(1, head_score))
        evaluation["feedback"]["head_position"] = [
            "Keep head steady and over the ball"
            if alignments and np.mean(alignments) > 50
            else "Good head position maintained",
            "Watch the ball closely through contact",
        ]

        # Swing Control score
        swing_score = 6.0
        if elbow_angles:
            good_elbow_ratio = sum(1 for a in elbow_angles if 100 <= a <= 140) / len(
                elbow_angles
            )
            swing_score += good_elbow_ratio * 4

        evaluation["scores"]["swing_control"] = min(10, max(1, swing_score))
        evaluation["feedback"]["swing_control"] = [
            "Maintain high elbow through the swing"
            if elbow_angles and np.mean(elbow_angles) < 110
            else "Good elbow position",
            "Follow through completely towards target",
        ]

        # Balance score
        balance_score = 6.5
        if spine_leans:
            good_lean_ratio = sum(1 for s in spine_leans if 5 <= s <= 25) / len(
                spine_leans
            )
            balance_score += good_lean_ratio * 3.5

        evaluation["scores"]["balance"] = min(10, max(1, balance_score))
        evaluation["feedback"]["balance"] = [
            "Keep spine angle consistent"
            if spine_leans and np.std(spine_leans) > 10
            else "Good balance throughout",
            "Transfer weight smoothly onto front foot",
        ]

        # Follow-through score (based on phase completion)
        followthrough_score = 6.0
        if "Follow-through" in phase_counts:
            followthrough_ratio = phase_counts["Follow-through"] / total_phases
            followthrough_score += followthrough_ratio * 20  # Weight this heavily

        evaluation["scores"]["follow_through"] = min(10, max(1, followthrough_score))
        evaluation["feedback"]["follow_through"] = [
            "Complete the follow-through motion"
            if followthrough_score < 7
            else "Good follow-through",
            "Finish with high hands and balanced position",
        ]

        # Summary statistics with improved consistency calculation
        consistency_score = 0.0
        if elbow_angles:
            # Calculate consistency based on variance of key metrics
            elbow_std = np.std(elbow_angles)
            elbow_consistency = max(
                0, 10 - (elbow_std / 10)
            )  # Normalize standard deviation

            spine_consistency = 10.0  # Default if no spine data
            if spine_leans:
                spine_std = np.std(spine_leans)
                spine_consistency = max(
                    0, 10 - (spine_std / 5)
                )  # Spine has smaller expected range

            alignment_consistency = 10.0  # Default if no alignment data
            if alignments:
                alignment_std = np.std(alignments)
                alignment_consistency = max(
                    0, 10 - (alignment_std / 20)
                )  # Normalize for pixel values

            # Weighted average of consistency scores
            consistency_score = (
                elbow_consistency * 0.5
                + spine_consistency * 0.3
                + alignment_consistency * 0.2
            )

        evaluation["summary_stats"] = {
            "avg_elbow_angle": np.mean(elbow_angles) if elbow_angles else None,
            "avg_spine_lean": np.mean(spine_leans) if spine_leans else None,
            "avg_alignment_distance": np.mean(alignments) if alignments else None,
            "consistency_score": round(consistency_score, 1),
            "elbow_std": round(np.std(elbow_angles), 1) if elbow_angles else None,
            "spine_std": round(np.std(spine_leans), 1) if spine_leans else None,
            "alignment_std": round(np.std(alignments), 1) if alignments else None,
        }

        # Overall score
        scores = list(evaluation["scores"].values())
        evaluation["overall_score"] = np.mean(scores)

        # Skill level prediction (bonus feature)
        overall = evaluation["overall_score"]
        if overall >= 8.5:
            evaluation["skill_level"] = "Advanced"
        elif overall >= 6.5:
            evaluation["skill_level"] = "Intermediate"
        else:
            evaluation["skill_level"] = "Beginner"

        return evaluation

    def generate_temporal_chart(self, output_dir: Path):
        """Generate temporal analysis chart (bonus feature)"""
        if not self.metrics_history:
            return

        timestamps = [m.timestamp for m in self.metrics_history]
        elbow_angles = [m.front_elbow_angle for m in self.metrics_history]
        spine_leans = [m.spine_lean for m in self.metrics_history]

        plt.figure(figsize=(12, 8))

        # Plot elbow angle
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, elbow_angles, label="Elbow Angle", color="blue", alpha=0.7)
        plt.fill_between(
            timestamps, 100, 140, alpha=0.2, color="green", label="Good Range"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Elbow Angle (degrees)")
        plt.title("Elbow Angle Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot spine lean
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, spine_leans, label="Spine Lean", color="red", alpha=0.7)
        plt.fill_between(
            timestamps, 5, 25, alpha=0.2, color="green", label="Good Range"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Spine Lean (degrees)")
        plt.title("Spine Lean Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / "temporal_analysis.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Temporal chart saved to: {chart_path}")


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description="Cricket Cover Drive Analysis")
    parser.add_argument("--url", type=str, help="YouTube URL to analyze")
    parser.add_argument("--video", type=str, help="Local video file to analyze")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    args = parser.parse_args()

    # Initialize analyzer
    config = {
        "output_dir": args.output,
        "thresholds": {
            "good_elbow_angle": (100, 140),
            "good_spine_lean": (5, 25),
            "good_alignment_threshold": 50,
            "good_foot_angle": (-30, 30),
        },
    }

    analyzer = CricketPoseAnalyzer(config)

    try:
        # Determine video source
        if args.url:
            video_path = analyzer.download_video(args.url)
        elif args.video:
            video_path = args.video
        else:
            # Default YouTube Short from requirements
            default_url = "https://youtube.com/shorts/vSX3IRxGnNY"
            print(f"No video specified, using default: {default_url}")
            video_path = analyzer.download_video(default_url)

        # Process video
        results = analyzer.process_video(video_path)

        # Add consistency analysis
        consistency_analysis = analyzer.analyze_consistency_issues()
        results["consistency_analysis"] = consistency_analysis

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"Output video: {results['output_video']}")
        print(f"Evaluation file: {results['evaluation_file']}")
        print(f"Processing FPS: {results['processing_stats']['average_fps']:.2f}")
        print(
            f"Target FPS achieved: {'✅' if results['processing_stats']['target_fps_achieved'] else '❌'}"
        )

        # Print consistency diagnosis
        print(f"\n🔍 CONSISTENCY ANALYSIS:")
        print(
            f"Valid detections - Elbow: {consistency_analysis['detection_rates']['elbow']:.1f}%, "
            f"Spine: {consistency_analysis['detection_rates']['spine']:.1f}%, "
            f"Alignment: {consistency_analysis['detection_rates']['alignment']:.1f}%"
        )

        if "elbow_stats" in consistency_analysis:
            elbow_stats = consistency_analysis["elbow_stats"]
            print(
                f"Elbow variation: {elbow_stats['std']:.1f}° (range: {elbow_stats['range']:.1f}°)"
            )

        # Print summary scores
        evaluation = results["evaluation"]
        print(f"\nOverall Score: {evaluation['overall_score']:.1f}/10")
        print(f"Skill Level: {evaluation['skill_level']}")
        print(
            f"Consistency Score: {evaluation['summary_stats']['consistency_score']:.1f}/10"
        )
        print("\nCategory Scores:")
        for category, score in evaluation["scores"].items():
            print(f"  {category.replace('_', ' ').title()}: {score:.1f}/10")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
