import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))

from functions import load_stats, filter_by_slot_used_source, slot_used_source_cats


def split(input_folder, output_folder, used_type=None):
    sall = load_stats(input_folder)[0]

    if used_type is not None:
        if used_type == "minimal":
            sall = filter_by_slot_used_source(sall, slot_used_source_cats.minimal)

        elif used_type == "minimal_patches":
            sall = filter_by_slot_used_source(
                sall, slot_used_source_cats.minimal_patches
            )
        else:
            raise ValueError(
                f"Unknown used type {used_type}\nValid is: minimal|minimal_patches"
            )

    os.makedirs(output_folder, exist_ok=True)

    for (program, program_set), usages in sall["used"].groupby(["name", "set"]):
        with open(
            os.path.join(output_folder, f"{program}_{program_set}_slots"), "w"
        ) as file:
            for closure, closure_usages in usages.groupby("closure"):
                file.write(closure)
                for slot, slot_usages in closure_usages.groupby("slot idx"):
                    if slot_usages.any():
                        file.write(",")
                        file.write(str(slot))
                file.write("\n")


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print(
            f"Unsupported number of arguments\nUsage:\n   {sys.argv[0]} input_folder output_folder [minimal|minimal_patches]"
        )
        sys.exit(1)

    try:
        split(
            input_folder=sys.argv[1],
            output_folder=sys.argv[2],
            used_type=sys.argv[3] if len(sys.argv) >= 4 else None,
        )
    except ValueError as e:
        print(e.args)
        sys.exit(1)
