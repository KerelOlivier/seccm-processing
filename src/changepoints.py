from tqdm import tqdm
from typing import Tuple

def find_changepoints(peaks) -> Tuple[list[int], list[int]]:
    touch_downs = []
    retractions = []
    prev_len = 0
    prev_val = -1

    # Smooth out peaks
    
    for i, peak in tqdm(enumerate(peaks)):
        if i > 20:
            if len(peak) > prev_len and (len(touch_downs) == 0 or i - touch_downs[-1] > 20):
                touch_downs.append(i)
            elif len(peak) >= 2:
                cur_val = peak[1]
                diff = cur_val - prev_val
                if prev_val != -1 and diff > 10:
                    retractions.append(i)
                prev_val = cur_val
            else:
                prev_val = -1
        prev_len = len(peak)
    return touch_downs, retractions
