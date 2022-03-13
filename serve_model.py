import argparse
import torch
from train_model import ModelCNN
import onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for exporting PyTorch model to ONNX with metadata")
    parser.add_argument("-hs", "--hash", required=True, type=str, help="Hash of commit")
    parser.add_argument("-d", "--date", required=True, type=str, help="Date of commit")
    parser.add_argument("-e", "--experiment", required=True, type=str, help="Name of experiment")

    args = parser.parse_args()

    model = ModelCNN()
    model.load_state_dict(torch.load("models/model_pytorch"))
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(model, dummy_input, "models/model.onnx",
                      input_names=["input"], output_names=["output"])

    onnx_model = onnx.load("models/model.onnx")
    onnx.checker.check_model(onnx_model)

    for pair in (("commit", args.hash), ("date", args.date), ("exp_name", args.experiment)):
        meta = onnx_model.metadata_props.add()
        meta.key, meta.value = pair
    onnx.save_model(onnx_model, "models/model.onnx")
