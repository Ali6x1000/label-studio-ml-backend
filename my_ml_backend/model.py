from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
import os
import json

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model integrating SAM2 for click→box predictions."""
    
    def setup(self):
        url = os.getenv("LABEL_STUDIO_URL")
        key = os.getenv("LABEL_STUDIO_API_KEY")
        print(f"LABEL_STUDIO_URL = {url!r}")
        print(f"LABEL_STUDIO_API_KEY = {key!r}")

        self.set("model_version", "1.0")

        checkpoint   = "/Users/alinawaf/Desktop/Lable-studio/label-studio-ml-backend/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        config_file  = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        print("Loading SAM2 model...")
        model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint,
            device=device,
            mode="eval",
            hydra_overrides_extra=[],
            apply_postprocessing=True,
        )
        print("SAM2 model loaded.")

        self.model = model
        self.predictor = SAM2ImagePredictor(
            sam_model=model,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        context = context or {}
        print(f">> predict() called with tasks={tasks}")
        print(f">> predict() context={context}")
        print(f">> predict() extra params={kwargs}")

        # load & parse JSON string of seen click IDs
        seen_raw = self.get("seen_click_ids") or "[]"
        seen_clicks = set(json.loads(seen_raw))
        print(f">> previously seen clicks: {seen_clicks}")

        predictions = []

        for task in tasks:
            print(f"\n-- processing task #{task['id']} --")
            # collect SAM trigger events
            events = context.get("result") or []
            if not events:
                ann = task.get("annotations", [])
                if ann:
                    events = ann[0].get("result", [])
                if not events:
                    drafts = task.get("drafts", [])
                    events = drafts[-1].get("result", []) if drafts else []
            print(f">> raw events: {events}")

            # filter to only new keypointlabels
            new_clicks = []
            for r in events:
                if r.get("type") == "keypointlabels" and r.get("from_name") == "click":
                    cid = r.get("id")
                    if cid not in seen_clicks:
                        new_clicks.append((cid, r["value"]))
            print(f">> new clicks: {new_clicks}")

            if not new_clicks:
                print(">> no new clicks, emitting empty result")
                predictions.append({
                    "result":        [],
                    "score":         None,
                    "model_version": self.get("model_version"),
                })
                continue

            # load image once
            img_path = self.get_local_path(task["data"]["image"], task_id=task["id"])
            img = np.array(Image.open(img_path))
            H, W = img.shape[:2]
            print(f">> image loaded ({W}×{H}) at {img_path}")
            self.predictor.set_image(img)

            boxes = []
            scores = []

            for cid, click in new_clicks:
                print(f">> processing click id={cid}, value={click}")
                x_px = click["x"] * W / 100.0
                y_px = click["y"] * H / 100.0
                print(f">> converted to pixels: ({x_px:.1f}, {y_px:.1f})")
                masks, sc, _ = self.predictor.predict(
                    point_coords=np.array([[x_px, y_px]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                mask = masks[0]
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    print(f">> raw box coords: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
                    box = {
                        "x":       100.0 * x0 / W,
                        "y":       100.0 * y0 / H,
                        "width":   100.0 * (x1 - x0) / W,
                        "height":  100.0 * (y1 - y0) / H,
                        "rectanglelabels": ["object"],
                    }
                    print(f">> box in % coords: {box}")
                    boxes.append({
                        "from_name": "label",
                        "to_name":   "image",
                        "type":      "rectanglelabels",
                        "value":     box
                    })
                    scores.append(float(sc[0]))
                else:
                    print(">> mask empty for that click")

                # mark click seen
                seen_clicks.add(cid)

            avg_score = sum(scores) / len(scores) if scores else None
            print(f">> aggregated {len(boxes)} boxes, avg_score={avg_score}")

            predictions.append({
                "result":        boxes,
                "score":         avg_score,
                "model_version": self.get("model_version"),
            })

        # persist back
        self.set("seen_click_ids", json.dumps(list(seen_clicks)))
        print(f">> updated seen_click_ids cache: {seen_clicks}")

        print(f"\n>> Predictions: {predictions}\n")
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        print(f">> fit() event={event}, data={data}")
        old_data = self.get('my_data')
        old_version = self.get('model_version')
        print(f">> old_data={old_data}, old_version={old_version}")
        self.set('my_data', json.dumps(data))
        self.set('model_version', '1.0.1')
        print(">> fit() completed successfully.")
