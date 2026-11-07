
def combine_text(data):
    return (
        data.fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
