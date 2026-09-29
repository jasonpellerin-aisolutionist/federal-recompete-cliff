import pandas as pd
import pytest

from recompete.score import check_not_shrunk


def test_shrink_guard_blocks_partial_data_and_allows_normal_weeks(tmp_path):
    published = tmp_path / "watchlist.parquet"
    pd.DataFrame({"piid": range(1000)}).to_parquet(published)
    check_not_shrunk(950, published)
    check_not_shrunk(10, tmp_path / "missing.parquet")
    with pytest.raises(SystemExit):
        check_not_shrunk(171, published)
