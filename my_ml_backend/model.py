from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
import os

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model integrating SAM2 for click→box predictions."""
    
    def setup(self):
        url = os.getenv("LABEL_STUDIO_URL")
        key = os.getenv("LABEL_STUDIO_API_KEY")

        print(f"LABEL_STUDIO_URL = {url!r}")
        print(f"LABEL_STUDIO_API_KEY = {key!r}")
        """Configure model and predictor."""
        self.set("model_version", "1.0")

        # paths to your config & checkpoint
        checkpoint = "/Users/alinawaf/Desktop/Lable-studio/label-studio-ml-backend/sam2/checkpoints/sam2.1_hiera_large.pt"
        config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"

        # choose CPU or MPS
        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        # build the SAM2 model
        model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint,
            device=device,
            mode="eval",
            hydra_overrides_extra=[],
            apply_postprocessing=True,
        )

        # wrap in the predictor
        self.model = model
        self.predictor = SAM2ImagePredictor(
            sam_model=model,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """On each click, run SAM2 and return one rectangle."""
        predictions = []
        context     = context or {}
        annotations = context.get("annotations", [])

        for idx, task in enumerate(tasks):
            # 1) try committed annotations
            ann = annotations[idx] if idx < len(annotations) else {}
            results = ann.get("result", [])

            # 2) fallback to latest draft if no committed
            if not results:
                drafts = task.get("drafts", [])
                if drafts:
                    results = drafts[-1].get("result", [])

            # 3) find the last keypoint click
            click = None
            for r in reversed(results):
                if r.get("type") == "keypointlabels" and r.get("from_name") == "click":
                    click = r["value"]
                    break

            # prepare default empty
            box_result, score = [], None

            if click:
                # load image at raw resolution
                img_path = self.get_local_path(task["data"]["image"], task_id=task["id"])
                img = np.array(Image.open(img_path))
                H, W = img.shape[:2]

                # set image once for SAM2
                self.predictor.set_image(img)

                # convert browser-% click → raw pixels
                x_px = click["x"] * W / 100.0
                y_px = click["y"] * H / 100.0

                # run SAM2 to get mask
                masks, scores, _ = self.predictor.predict(
                    point_coords   = np.array([[x_px, y_px]]),
                    point_labels   = np.array([1]),
                    multimask_output=False,
                )
                mask = masks[0]

                # extract tight box in raw pixels
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()

                    # normalize back to browser-%
                    box = {
                        "x":       100.0 * (x0 / W),
                        "y":       100.0 * (y0 / H),
                        "width":   100.0 * ((x1 - x0) / W),
                        "height":  100.0 * ((y1 - y0) / H),
                        "rectanglelabels": ["object"],
                    }
                    box_result = [{
                        "from_name": "label",
                        "to_name":   "image",
                        "type":      "rectanglelabels",
                        "value":     box
                    }]
                    score = float(scores[0])

            predictions.append({
                "result":        box_result,
                "score":         score,
                "model_version": self.get("model_version"),
            })

        return ModelResponse(predictions=predictions)




    def fit(self, event, data, **kwargs):
        """Optional: handle incremental training on annotation events."""
        old_data = self.get('my_data')
        old_version = self.get('model_version')
        print(f'Old data: {old_data}, version: {old_version}')

        # example of caching new info
        self.set('my_data', data)
        self.set('model_version', '0.0.2')
        print('fit() completed successfully.')
