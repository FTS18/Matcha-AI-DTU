# Matcha AI System Architecture

## Overview
Matcha AI is a distributed intelligence system designed for automated sports analytics, specifically optimized for football (soccer). The system utilizes a multi-service architecture to process video streams, perform real-time object tracking, and synthesize visual narratives.

## Core Components

### 1. Inference Engine (`services/inference`)
The inference engine is the computational core of the system. It has been refactored into a modular sub-package structure to ensure high cohesion and low coupling.

*   **Video Processing (`app.core.video`)**: Handles frame-accurate extraction and synchronization using specialized codecs.
*   **Soccer Analytics (`app.core.soccer_analysis`)**: Implements the primary detection logic. It utilizes a Kalman-filtered tracking system for the ball and players, with specialized heuristics for goal event detection.
*   **Visual Synthesis (`app.core.visuals`)**: Translates raw coordinate data into human-readable overlays, including heatmaps, speed vectors, and highlight reels.

### 2. Orchestrator Service (`services/orchestrator`)
The orchestrator serves as the primary API gateway and state manager. It coordinates the lifecycle of an "Analysis Job":
*   **Ingestion**: Receives video uploads and validates file integrity.
*   **Task Dispatching**: Communicates with the inference microservices via a stub-based routing system.
*   **Persistence**: Manages the storage of tracking data and generated highlights.

### 3. Web Dashboard (`apps/web`)
A Next.js 14-based front-end that provides a high-performance interface for viewing analytics. It utilizes WebSocket connections for real-time status updates during long-running inference tasks.

## Data Flow Pipeline
1.  **Ingestion**: User uploads MP4/MOV via the Web Dashboard.
2.  **Normalization**: Orchestrator secures the file and assigns a unique Analysis ID.
3.  **Inference**: The engine processes frames, generating a temporal coordinate map.
4.  **Narrative Generation**: LLM-based logic (in `routes_stub.py`) synthesizes the tracking data into a descriptive match narrative.
5.  **Visualization**: Final visual assets are rendered and served back to the dashboard.
