def clean_column (df):
  df.columns = (
    df.columns
    .str.strip()                   # remove leading/trailing spaces
    .str.lower()                   # make lowercase
    .str.replace(r'[^\w]', '_', regex=True)  # replace non-alphanumeric characters with _
  )
  return df
