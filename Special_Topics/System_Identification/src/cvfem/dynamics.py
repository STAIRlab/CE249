"""
From video, extract motion data for dynamic analysis.
"""
import cv2
import numpy as np

def get_video_frames(video_path, stabilize=False):
    """
    Extract frames from a video.
    """

    # Initialize video capture
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []

    if stabilize:
        prev_frame = None # Initial previous frame is None
        prev_frame_gray = None # Initial previous gray frame is None
        prev_x, prev_y = 0, 0  # Initial offset is zero
        while capture.isOpened:
            ret, frame = capture.read()
            if not ret:
                break  # End of video
        
            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate optical flow (tracking feature points)
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Calculate the transformation matrix (translation)
                dx = np.mean(flow[...,0])
                dy = np.mean(flow[...,1])
                # Translate the frame to compensate for the previous motion
                M = np.float32([[1, 0, dx - prev_x], [0, 1, dy - prev_y]])
                stabilized_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                frames.append(stabilized_frame)
                prev_x, prev_y = dx, dy

            prev_frame = frame
            prev_frame_gray = frame_gray

    else:
        while capture.isOpened:
            ret, frame = capture.read()
            if not ret:
                break  # End of video
            frames.append(frame)

    capture.release()
    return frames, fps


def track_object(video_path, stabilize=False, n_roi=1, roi=None):
    """
    Track an object on a video.
    """
    
    frames, fps = get_video_frames(video_path, stabilize=stabilize)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames to track.")
    # Select the first frame for tracking
    first_frame = frames[0]
    # Check if frame is empty or invalid
    if first_frame is None or first_frame.size == 0:
        raise ValueError("First frame is empty or invalid.")
    
    # If ROI (Region Of Interest) is provided, use it; else manually select it
    if roi is None:
        roi = []
        for i in range(n_roi):
            r = cv2.selectROI(first_frame)
            roi.append(r)
            x, y, w, h = r
            # Ensure ROI is valid
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError(f"Invalid Range of Interest (ROI): {roi}")
            # All these waitkeys are a hack to get the OpenCV window to close
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            for _ in range (1,5):
                cv2.waitKey(1)

    # Set up the tracker (e.g., CSRT or MOSSE)
    # tracker = cv2.TrackerKCF_create() # KCF tracker (more robust with different frame types)
    tracker = cv2.TrackerCSRT_create()  # Use the CSRT tracker (non-legacy)
    # tracker = cv2.legacy.TrackerMOSSE_create()  # Use the MOSSE tracker (legacy)
    
    positions = np.empty((n_frames,n_roi,2))
    for i,r in enumerate(roi):
        tracker.init(first_frame, r)
        # Track the object across all frames
        for j,frame in enumerate(frames):
            # Convert each frame to grayscale (single-channel) for tracker update
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            ret, bbox = tracker.update(frame_gray)
            if ret:
                x, y, w, h = [int(v) for v in bbox]
                positions[j,i] = (x + w/2, y + h/2)  # Store center of the object
    
    return positions, fps


# Differentiating from region positions to acceleration
def pos_to_accel(positions, fps, center=True, scale_factor=1):
    if center:
        positions = positions-np.mean(positions[:100], axis=0)
    displ = positions*scale_factor
    veloc = np.diff(displ,axis=0)/fps
    accel = np.diff(veloc,axis=0)/fps
    return displ, veloc, accel


# Integrating from acceleration to displacement
def accel_to_displ(accel, dt=1, method='cumsum'):
    # Depending on the chosen method, integrate
    # twice to get displacement.
    if method == 'cumsum':
        veloc = dt*np.cumsum(accel)
        displ = dt*np.cumsum(veloc)
    elif method == 'cumtrapz':
        import scipy.integrate as spint
        veloc = spint.cumulative_trapezoid(accel, dx=dt)
        displ = spint.cumulative_trapezoid(veloc, dx=dt)
    return displ, veloc, accel


# Plotting function for displacement, velocity, and acceleration
def plot_results(time, displ, veloc, accel, dim=0):
    from matplotlib import pyplot as plt
    n_frames,n_roi,n_dim = displ.shape
    fig,ax = plt.subplots(3,1, figsize=(8, 5), constrained_layout=True)
    ax[0].plot(time, displ[:,:,dim], label=[f"Floor {i}" for i in range(n_roi)])
    ax[0].set_title('Displacement (cm)')
    ax[1].plot(time[1:], veloc[:,:,dim], label=[f"Floor {i}" for i in range(n_roi)])
    ax[1].set_title('Velocity (cm/s)')
    ax[2].plot(time[2:], accel[:,:,dim], label=[f"Floor {i}" for i in range(n_roi)])
    ax[2].set_title('Acceleration (cm/s/s)')
    for axi in ax:
        axi.legend(bbox_to_anchor=(1.2,1))
    return fig