from export_model import split_grouped_files


def test_split_grouped_files_keeps_unique_files_and_respects_holdout_sizes():
    files_by_group = {
        "Normal_0Nm": ["Normal_0Nm_01.mat", "Normal_0Nm_02.mat", "Normal_0Nm_03.mat", "Normal_0Nm_04.mat", "Normal_0Nm_05.mat"],
        "BPFI_2Nm": ["BPFI_2Nm_01.mat", "BPFI_2Nm_02.mat", "BPFI_2Nm_03.mat", "BPFI_2Nm_04.mat", "BPFI_2Nm_05.mat"],
        "Unbalance_4Nm": ["Unbalance_4Nm_01.mat", "Unbalance_4Nm_02.mat", "Unbalance_4Nm_03.mat", "Unbalance_4Nm_04.mat", "Unbalance_4Nm_05.mat"],
    }

    train_files, val_files, test_files = split_grouped_files(files_by_group, seed=42)

    all_files = set(train_files) | set(val_files) | set(test_files)
    assert len(all_files) == 15
    assert set(train_files).isdisjoint(val_files)
    assert set(train_files).isdisjoint(test_files)
    assert set(val_files).isdisjoint(test_files)
    assert len(train_files) >= 7
    assert len(val_files) >= 3
    assert len(test_files) >= 3


def test_split_grouped_files_works_for_small_groups_without_duplicating_files():
    files_by_group = {"BPFO_0Nm": ["BPFO_0Nm_01.mat", "BPFO_0Nm_02.mat", "BPFO_0Nm_03.mat"]}

    train_files, val_files, test_files = split_grouped_files(files_by_group, seed=7)

    all_files = set(train_files) | set(val_files) | set(test_files)
    assert len(all_files) == 3
    assert len(train_files) + len(val_files) + len(test_files) == 3
