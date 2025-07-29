import torch
import argparse

ignore_keys = ["num_batches_tracked", "running_mean", "running_var"]


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if any([key in key_item_1[0] for key in ignore_keys]):
            continue
        elif torch.all(torch.isclose(key_item_1[1], key_item_2[1]) == True):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


def load_model(model_path):
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two models")
    parser.add_argument(
        "-m1", "--model_1", type=str, help="Path to model 1", required=True
    )
    parser.add_argument(
        "-m2", "--model_2", type=str, help="Path to model 2", required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_1 = load_model(args.model_1)
    model_2 = load_model(args.model_2)
    compare_models(model_1, model_2)
