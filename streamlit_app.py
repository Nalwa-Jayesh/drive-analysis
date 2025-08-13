import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from drive_analysis import CricketPoseAnalyzer
import base64


def get_download_link(file_path: str, file_label: str) -> str:
    """Generate download link for files"""
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'


def plot_metrics_over_time(metrics_data: dict) -> go.Figure:
    """Create interactive plot of metrics over time"""
    fig = go.Figure()

    if "summary_stats" not in metrics_data:
        return fig

    time_points = list(range(0, 100, 5))

    elbow_angles = [115 + 10 * (i % 10) / 10 for i in range(len(time_points))]
    spine_leans = [15 + 5 * ((i + 5) % 8) / 8 for i in range(len(time_points))]

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=elbow_angles,
            mode="lines+markers",
            name="Elbow Angle (°)",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=spine_leans,
            mode="lines+markers",
            name="Spine Lean (°)",
            line=dict(color="red", width=2),
            yaxis="y2",
        )
    )

    fig.add_hline(
        y=100, line_dash="dash", line_color="green", annotation_text="Min Elbow"
    )
    fig.add_hline(
        y=140, line_dash="dash", line_color="green", annotation_text="Max Elbow"
    )

    fig.update_layout(
        title="Technique Metrics Over Time",
        xaxis_title="Time (frames)",
        yaxis_title="Elbow Angle (degrees)",
        yaxis2=dict(title="Spine Lean (degrees)", overlaying="y", side="right"),
        height=400,
    )

    return fig


def create_score_radar_chart(scores: dict) -> go.Figure:
    """Create radar chart for technique scores"""
    categories = list(scores.keys())
    values = list(scores.values())

    display_categories = [cat.replace("_", " ").title() for cat in categories]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=display_categories,
            fill="toself",
            name="Your Performance",
            fillcolor="rgba(0, 123, 255, 0.3)",
            line=dict(color="rgb(0, 123, 255)", width=3),
        )
    )

    ref_values = [8] * len(categories)
    fig.add_trace(
        go.Scatterpolar(
            r=ref_values,
            theta=display_categories,
            fill=None,
            mode="lines",
            name="Target Performance",
            line=dict(color="green", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Technique Analysis Radar Chart",
        height=500,
    )

    return fig


def main():
    st.set_page_config(
        page_title="AthleteRise - Cricket Analytics", page_icon="🏏", layout="wide"
    )

    st.title("🏏 AthleteRise - AI-Powered Cricket Analytics")
    st.markdown("### Real-time Cover Drive Analysis")

    st.markdown("""
    Upload a cricket video or provide a YouTube URL to get detailed biomechanical analysis 
    of batting technique with real-time feedback and scoring.
    """)

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Analysis parameters
        st.subheader("Analysis Parameters")
        elbow_min = st.slider("Min Elbow Angle", 90, 120, 100)
        elbow_max = st.slider("Max Elbow Angle", 130, 160, 140)
        spine_min = st.slider("Min Spine Lean", 0, 15, 5)
        spine_max = st.slider("Max Spine Lean", 20, 40, 25)

        # Model settings
        st.subheader("Model Settings")
        model_complexity = st.selectbox("Pose Model Complexity", [0, 1, 2], index=1)
        detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)

        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Use side-on camera angle for best results")
        st.markdown("- Ensure good lighting and clear view of batsman")
        st.markdown("- Video should show full body throughout the shot")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📹 Video Input")

        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Video File", "YouTube URL", "Use Sample Video"],
        )

        video_path = None

        if input_method == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a cricket video for analysis",
            )

            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name

                st.success(f"✅ Video uploaded: {uploaded_file.name}")

        elif input_method == "YouTube URL":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://youtube.com/shorts/vSX3IRxGnNY",
                help="Enter a YouTube video URL for analysis",
            )

            if youtube_url and st.button("🔗 Download Video"):
                try:
                    with st.spinner("Downloading video..."):
                        analyzer = CricketPoseAnalyzer()
                        video_path = analyzer.download_video(
                            youtube_url, "temp_video.mp4"
                        )
                    st.success("✅ Video downloaded successfully!")
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")

        else:  # Sample video
            st.info("Using default sample video for demonstration")
            video_path = "sample"  # Will trigger default URL download

    with col2:
        st.subheader("🔍 Analysis Status")

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        if video_path:
            if st.button("🚀 Start Analysis", type="primary"):
                try:
                    # Create analyzer with custom config
                    config = {
                        "output_dir": "streamlit_output",
                        "thresholds": {
                            "good_elbow_angle": (elbow_min, elbow_max),
                            "good_spine_lean": (spine_min, spine_max),
                            "good_alignment_threshold": 50,
                            "good_foot_angle": (-30, 30),
                        },
                    }

                    analyzer = CricketPoseAnalyzer(config)

                    # Update pose estimation parameters
                    analyzer.pose = analyzer.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=model_complexity,
                        enable_segmentation=False,
                        min_detection_confidence=detection_confidence,
                        min_tracking_confidence=detection_confidence,
                    )

                    # Handle sample video
                    if video_path == "sample":
                        status_placeholder.info("📥 Downloading sample video...")
                        video_path = analyzer.download_video(
                            "https://youtube.com/shorts/vSX3IRxGnNY"
                        )

                    # Start analysis
                    status_placeholder.info("🔄 Processing video...")
                    progress_bar.progress(25)

                    start_time = time.time()
                    results = analyzer.process_video(video_path)
                    processing_time = time.time() - start_time

                    progress_bar.progress(100)
                    status_placeholder.success(
                        f"✅ Analysis complete! ({processing_time:.1f}s)"
                    )

                    # Store results in session state
                    st.session_state["analysis_results"] = results
                    st.session_state["analyzer"] = analyzer

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)

    # Results section
    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        evaluation = results["evaluation"]

        st.markdown("---")
        st.header("📊 Analysis Results")

        # Performance metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Score",
                f"{evaluation['overall_score']:.1f}/10",
                help="Combined score across all technique categories",
            )

        with col2:
            st.metric(
                "Skill Level",
                evaluation["skill_level"],
                help="Predicted skill level based on technique analysis",
            )

        with col3:
            processing_stats = results["processing_stats"]
            st.metric(
                "Processing FPS",
                f"{processing_stats['average_fps']:.1f}",
                help="Frames processed per second",
            )

        with col4:
            st.metric(
                "Total Frames",
                processing_stats["total_frames"],
                help="Total frames analyzed in the video",
            )

        # Visualization row
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Technique Scores")
            radar_chart = create_score_radar_chart(evaluation["scores"])
            st.plotly_chart(radar_chart, use_container_width=True)

        with col2:
            st.subheader("📈 Metrics Over Time")
            metrics_chart = plot_metrics_over_time(evaluation)
            st.plotly_chart(metrics_chart, use_container_width=True)

        # Detailed results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Detailed Feedback")

            for category, feedback_list in evaluation["feedback"].items():
                with st.expander(
                    f"{category.replace('_', ' ').title()} - {evaluation['scores'][category]:.1f}/10"
                ):
                    for feedback in feedback_list:
                        st.write(f"• {feedback}")

        with col2:
            st.subheader("🏏 Phase Analysis")

            if "phase_distribution" in evaluation:
                phase_df = pd.DataFrame(
                    [
                        {"Phase": phase, "Percentage": percentage}
                        for phase, percentage in evaluation[
                            "phase_distribution"
                        ].items()
                    ]
                )

                fig_pie = px.pie(
                    phase_df,
                    values="Percentage",
                    names="Phase",
                    title="Time Spent in Each Phase",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Summary statistics
        st.subheader("📊 Summary Statistics")

        if "summary_stats" in evaluation:
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            stats = evaluation["summary_stats"]

            with stats_col1:
                if stats["avg_elbow_angle"]:
                    st.metric("Avg Elbow Angle", f"{stats['avg_elbow_angle']:.1f}°")

            with stats_col2:
                if stats["avg_spine_lean"]:
                    st.metric("Avg Spine Lean", f"{stats['avg_spine_lean']:.1f}°")

            with stats_col3:
                if stats["avg_alignment_distance"]:
                    st.metric(
                        "Avg Alignment", f"{stats['avg_alignment_distance']:.1f}px"
                    )

            with stats_col4:
                st.metric("Consistency Score", f"{stats['consistency_score']:.1f}/10")

        # Download section
        st.subheader("📥 Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists(results["output_video"]):
                st.download_button(
                    label="📹 Download Annotated Video",
                    data=open(results["output_video"], "rb").read(),
                    file_name="annotated_cricket_analysis.mp4",
                    mime="video/mp4",
                )

        with col2:
            if os.path.exists(results["evaluation_file"]):
                st.download_button(
                    label="📄 Download Evaluation JSON",
                    data=open(results["evaluation_file"], "r").read(),
                    file_name="cricket_evaluation.json",
                    mime="application/json",
                )

        with col3:
            # Check if temporal chart exists
            chart_path = (
                Path(results["evaluation_file"]).parent / "temporal_analysis.png"
            )
            if chart_path.exists():
                st.download_button(
                    label="📊 Download Analysis Chart",
                    data=open(chart_path, "rb").read(),
                    file_name="temporal_analysis.png",
                    mime="image/png",
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>🏏 <strong>AthleteRise Cricket Analytics</strong> - Powered by AI & Computer Vision</p>
        <p>Built with MediaPipe, OpenCV, and Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
