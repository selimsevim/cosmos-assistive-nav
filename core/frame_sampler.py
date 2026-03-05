from utils.image_utils import preprocess_frame

def sample_frames(frame_a, frame_b):
    """
    Processes two chronological frames for API submission.
    frame_a: previous frame (1 second ago)
    frame_b: current frame
    
    Returns standard Base64 string for both frames.
    """
    frame_a_b64 = preprocess_frame(frame_a)
    frame_b_b64 = preprocess_frame(frame_b)
    
    return frame_a_b64, frame_b_b64
