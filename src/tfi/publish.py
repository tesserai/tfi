import decimal
import hashlib
import json
import requests
import tempfile
import uuid

from tqdm import tqdm
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor


def sha256_for_file(f, buf_size=65536):
    pos = f.tell()
    dgst = hashlib.sha256()
    while True:
        data = f.read(buf_size)
        if not data:
            break
        dgst.update(data)
    size = f.tell() - pos
    f.seek(pos)

    return size, dgst.hexdigest()

environment_name = "tfi"
namespace = "default"
environment = {
    "namespace": namespace,
    "name": environment_name,
}

fission_url = "http://35.202.47.203"
def post(rel_url, data):
    response = requests.post(
            "%s%s" % (fission_url, rel_url),
            data=json.dumps(data),
            headers={"Content-Type": "application/json"})
    # print("POST", rel_url)
    # print(response, response.text)
    if response.status_code in [404, 409]:
        return response.status_code, None
    if response.status_code == 500:
        raise Exception(response.text)
    return response.status_code, response.json()

def get(rel_url, params=None):
    response = requests.get(
            "%s%s" % (fission_url, rel_url),
            params=params)
    if response.status_code == 404:
        return response.status_code, None
    if response.status_code == 500:
        raise Exception(response.text)
    return response.status_code, response.json()

def format_bytes(count):
    label_ix = 0
    labels = ["B", "KiB", "MiB", "GiB"]
    while label_ix < len(labels) and count / 1024. > 1:
        count = count / 1024.
        label_ix += 1
    count = decimal.Decimal(count)
    count = count.to_integral() if count == count.to_integral() else round(count.normalize(), 2)
    return "%s %s" % (count, labels[label_ix])

def lazily_define_package(file):
    filesize, archive_sha256 = sha256_for_file(file)
    base_archive_url = "%s/proxy/storage/v1/archive" % fission_url

    status_code, response = get("/v2/packages/%s" % archive_sha256)
    if status_code == 200:
        print("Already uploaded", flush=True)
        return archive_sha256, response

    progress = tqdm(
            total=filesize,
            desc="Uploading",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True)

    last_bytes_read = 0
    def update_progress(monitor):
        # Your callback function
        nonlocal last_bytes_read
        progress.update(monitor.bytes_read - last_bytes_read)
        last_bytes_read = monitor.bytes_read

    e = MultipartEncoder(fields={'uploadfile': ('uploaded', file, 'text/plain')})
    m = MultipartEncoderMonitor(e, update_progress)
    archive_response = requests.post(base_archive_url,
            data=m,
            headers={
                "X-File-Size": str(filesize),
                'Content-Type': m.content_type})

    archive_id = archive_response.json()['id']
    print(" done", flush=True)

    archive_url = "%s?id=%s" % (base_archive_url, archive_id)

    package = {
        "metadata": {
            "name": archive_sha256,
            "namespace": namespace,
        },
        "spec": {
            "environment": environment,
            "deployment": {
                "type": "url",
                "url": archive_url,
                "checksum": {
                        "type": "sha256",
                        "sum": archive_sha256,
                },
            },
        },
        "status": {
            "buildstatus": "succeeded",
        },
    }
    return archive_sha256, post("/v2/packages", package)[1]

def lazily_define_function(f):
    archive_sha256, package_ref = lazily_define_package(f)
    print("Registering ...", end='', flush=True)
    function_name = archive_sha256[:8]
    status_code, response = get("/v2/functions/%s" % function_name)
    if status_code == 200:
        return function_name

    status_code, r = post("/v2/functions", {
        "metadata": {
            "name": function_name,
            "namespace": namespace,
        },
        "spec": {
            "environment": environment,
            "package": {
                "functionName": function_name,
                "packageref": package_ref,
            },
        },
    })
    if status_code == 409 or status_code == 201:
        print(" done", flush=True)
        return function_name

    print(" error", flush=True)
    raise Exception(r.text)

def lazily_define_trigger2(function_name, http_method, host, relativeurl):
    trigger_name = "%s-%s-%s" % (
            host.replace('.', '-'),
            relativeurl.replace(':.*', '').replace('{', '').replace('}', '').replace('/', '-'),
            http_method.lower())
    status_code, response = get("/v2/triggers/http/%s" % trigger_name)
    if status_code == 200:
        return

    status_code, r = post("/v2/triggers/http", {
        "metadata": {
            "name":      trigger_name,
            "namespace": namespace,
        },
        "spec": {
            "host": host,
            "relativeurl": relativeurl,
            "method":      http_method,
            "functionref": {
                "Type": "name",
                "Name": function_name,
            },
        },
    })
    if status_code == 409 or status_code == 201:
        return
    raise Exception(r.text)

def publish(f):
    function_name = lazily_define_function(f)
    host = "%s.tfi.gcp.tesserai.com" % function_name
    lazily_define_trigger2(function_name, "POST", host, "/{path-info:.*}")
    lazily_define_trigger2(function_name, "GET", host, "/{path-info:.*}")
    lazily_define_trigger2(function_name, "GET", host, "/")
    return "http://%s" % host

