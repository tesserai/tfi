import inspect
import torch

deserialized_objects = {}
restore_location = torch.serialization.default_restore_location
def _check_container_source(container_type, source_file, original_source):
    current_source = inspect.getsource(container_type)
    if original_source != current_source:
        if container_type.dump_patches:
            file_name = container_type.__name__ + '.patch'
            diff = difflib.unified_diff(current_source.split('\n'),
                                        original_source.split('\n'),
                                        source_file,
                                        source_file, lineterm="")
            lines = '\n'.join(diff)
            try:
                with open(file_name, 'a+') as f:
                    file_size = f.seek(0, 2)
                    f.seek(0)
                    if file_size == 0:
                        f.write(lines)
                    elif file_size != len(lines) or f.read() != lines:
                        raise IOError
                msg = ("Saved a reverse patch to " + file_name + ". "
                       "Run `patch -p0 < " + file_name + "` to revert your "
                       "changes.")
            except IOError:
                msg = ("Tried to save a patch, but couldn't create a "
                       "writable file " + file_name + ". Make sure it "
                       "doesn't exist and your working directory is "
                       "writable.")
        else:
            msg = ("you can retrieve the original source code by "
                   "accessing the object's source attribute or set "
                   "`torch.nn.Module.dump_patches = True` and use the "
                   "patch tool to revert the changes.")
        msg = ("source code of class '{}' has changed. {}"
               .format(torch.typename(container_type), msg))
        warnings.warn(msg, SourceChangeWarning)

def persistent_load(saved_id):
    assert isinstance(saved_id, tuple)
    typename = saved_id[0]
    data = saved_id[1:]

    if typename == 'module':
        # Ignore containers that don't have any sources saved
        if all(data[1:]):
            _check_container_source(*data)
        return data[0]
    elif typename == 'storage':
        data_type, root_key, location, size, view_metadata = data
        if root_key not in deserialized_objects:
            deserialized_objects[root_key] = restore_location(
                data_type(size), location)
        storage = deserialized_objects[root_key]
        if view_metadata is not None:
            view_key, offset, view_size = view_metadata
            if view_key not in deserialized_objects:
                deserialized_objects[view_key] = storage[offset:offset + view_size]
            return deserialized_objects[view_key]
        else:
            return storage
    else:
        raise RuntimeError("Unknown saved id type: %s" % saved_id[0])
