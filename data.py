import pandera as pa


class TSDataSchema(pa.DataFrameModel):
    """Schema for input, follows the same format as Nixtla."""

    unique_id: pa.typing.Series[pa.String]
    ds: pa.typing.Series[pa.typing.Object]
    y: pa.typing.Series[pa.Float]
