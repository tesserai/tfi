def asset_path(model, asset_name):
  if not hasattr(model, '__tfi_asset_path__'):
    return None
  return model.__tfi_asset_path__(asset_name)
