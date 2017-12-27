## For TF 1.0
```
cd ~/github/tesserai/tfi-models
source ./venv/bin/activate
export PYTHONPATH=$PWD:/Users/adamb/github/tesserai/tfi/src/:$PWD/dep/tensorflow/models/slim:$PYTHONPATH
export PATH=$PWD/../tfi/src/tfi/:$PATH
```

## Look at model definition

## Use model directly from the command line
```
tfi \
  src.tensorflow.magenta.image_stylization.MagentaImageStylize \
  --checkpoint_file="./src/tensorflow/magenta/checkpoints/image_stylization/multistyle-pastiche-generator-varied.ckpt" \
  --num_styles=32 -- \
   'stylize(style_weights=[0, 0, 0, 1, *([0] * 28)])' \
   --images=@/Users/adamb/Desktop/dog-medium-landing-hero.jpg
```

## Mess with model interactively...
```
# Launch interactive console
tfi \
  src.tensorflow.magenta.image_stylization.MagentaImageStylize \
  --checkpoint_file="./src/tensorflow/magenta/checkpoints/image_stylization/multistyle-pastiche-generator-varied.ckpt" \
  --num_styles=32
```


# Look at input
```
dog = tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg")
```

# Try a random style
```
stylize(
    images=[dog],
    style_weights=[1 if ix == 27 else 0 for ix in range(n_styles)]).images[0]
    ```

# Try all the styles
```
stylized = [
    stylize(
        images=[tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg")],
        style_weights=[1 if ix == style_no else 0 for ix in range(hparams().num_styles)]).images[0]
    for style_no in range(hparams().num_styles)
]
```

# Look at some of them...
```
stylized[1:3]
```


########
# For saving and exporting...
########

## For TF 1.4
```
# Set up from scratch...
virtualenv venv-torch
./venv-tf-1.4/bin/pip3 install magenta tensorflow numpy ptpython six pygments prompt_toolkit tinydb

###

cd ~/github/tesserai/tfi-models
source ./venv-tf-1.4/bin/activate
export PYTHONPATH=$PWD:/Users/adamb/github/tesserai/tfi/src/:$PWD/dep/tensorflow/models/slim:$PYTHONPATH
export PATH=$PWD/../tfi/src/tfi/:$PATH
```


# Try Inception V1
```
tfi src.tensorflow.models.slim.Model --name="inception_v1"
```

```
image = tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg")
result = predict(images=[image])
categories, scores = result.categories, result.scores[0]

(scores[scores.argsort()[-1]], categories[scores.argsort()[-1]].decode())
```

# Export to a saved_model
```
tfi src.tensorflow.models.slim.Model --name="inception_v1" --export foo-inception-v1.saved_model
```

```
tfi @foo-inception-v1.saved_model
```



#######
# For pytorch

```
# Set up from scratch...
virtualenv venv-torch
./venv-torch/bin/pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl
./venv-torch/bin/pip3 install numpy ptpython six pygments prompt_toolkit tinydb torchvision



###

cd ~/github/tesserai/tfi-models
source ./conda/bin/activate
export PYTHONPATH=$PWD:/Users/adamb/github/tesserai/tfi/src/:$PWD/dep/tensorflow/models/slim:$PYTHONPATH
export PATH=$PWD/../tfi/src/tfi/:$PATH
cd ../tfi
```

#

r = predict(images=[tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg")])
categories, scores = r['categories'], r['scores'][0].sort()

(scores[0][-1], categories[scores[1][-1].data[0]])


r = predict(images=[tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg"), tfi.data.file("/Users/adamb/Desktop/1473094579048.jpg")])

r = predict(images=[tfi.data.file("/Users/adamb/Desktop/1473094579048.jpg"), tfi.data.file("/Users/adamb/Desktop/1473094579048.jpg")])

r = predict(images=[tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg"), tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg"), tfi.data.file("/Users/adamb/Desktop/1473094579048.jpg")])

r = predict(images=[tfi.data.file("/Users/adamb/Desktop/dog-medium-landing-hero.jpg"), tfi.data.file("/Users/adamb/Desktop/1473094579048.jpg")])
categories, scores = r['categories'], r['scores']

[
  [(categories[ix.data[0]], score.data[0]) for score, ix in zip(*scorez.topk(5, 0, True, True))]
  for scorez in scores
]



source /opt/miniconda/bin/activate someenv

python

import urllib.request
from urllib.request import urlopen
urllib.request.ProxyHandler(proxies={'http': 'http://10.128.0.2:1', 'https': 'http://10.128.0.2:1'})
len(urlopen("http://download.pytorch.org/models/resnet50-19c8e357.pth").read())

urllib.request.getproxies()

kubectl exec squid-0nf62 -c squid -- sh -c 'tail -f /var/log/squid3/*'

ip route change $(ip route show | grep default) proto static initrwnd 20


kubectl exec  tfi-resnet50-demo-3962754815-wnn4h -t -i -- bash -li
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTP_PROXY
wget http://download.pytorch.org/models/resnet50-19c8e357.pth


securityContext:
  capabilities:
    add:
      - NET_ADMIN


cd ~/github/tesserai/deployment
nix-shell -I nixpkgs=https://github.com/NixOS/nixpkgs-channels/archive/96457d26dded05bcba8e9fbb9bf0255596654aab.tar.gz

# Update fission
(cd gopath/src/github.com/fission/fission/fission && go install)


## Push TFI environment
```
(
VERSION=0.22
GCLOUD_PROJECT=just-shape-151123
DOCKER_TAG=gcr.io/$GCLOUD_PROJECT/tfi-env:$VERSION
docker build -t $DOCKER_TAG .
docker run -v /Users/adamb/github/tesserai/tfi:/Users/adamb/github/tesserai/tfi -p "8080:8888" $DOCKER_TAG
)

curl -H "Content-Type: application/json" --data-ascii '{"filepath":"/Users/adamb/github/tesserai/tfi/resnet50.tfi", "functionName": "topk"}' http://127.0.0.1:8080/v2/specialize
time curl -F k=2 -F "images=@/Users/adamb/Desktop/dog-medium-landing-hero.png;type=image/png" http://127.0.0.1:8080/

# Prepare fission deployment
export FISSION_ROUTER=$(kubectl --namespace fission get svc router -o=jsonpath='{..ip}')
export FISSION_URL=http://$(kubectl --namespace fission get svc controller -o=jsonpath='{..ip}')
(
pushd ../tfi
VERSION=0.33
TFI_ENV_IMAGE=gcr.io/$GCLOUD_PROJECT/tfi-env:$VERSION
docker build -t $TFI_ENV_IMAGE .
gcloud docker -- push $TFI_ENV_IMAGE
fission env update --name tfi --image $TFI_ENV_IMAGE
)

curl -F k=2 -F "images=@/Users/adamb/Desktop/dog-medium-landing-hero.png;type=image/png" http://$FISSION_ROUTER/topk

fission function update --name topk --env gcr.io/$GCLOUD_PROJECT/tfi-env:$VERSION

```

helm install --set "image=$DOCKER_TAG,pullPolicy=IfNotPresent,analytics=false" charts/fission-all

## To debug a helm template:
## - comment out the yaml that won't parse
## Run helm install --dry-run --debug ...

(
set -ex
VERSION=0.4.4
FISSION_IMAGE=gcr.io/$GCLOUD_PROJECT/fission-bundle
FISSION_IMAGE_FQN=$FISSION_IMAGE:$VERSION
FISSION_FLUENTD_IMAGE=gcr.io/$GCLOUD_PROJECT/fission-fluentd
FISSION_FLUENTD_IMAGE_FQN=$FISSION_FLUENTD_IMAGE:$VERSION
pushd $GOPATH/src/github.com/fission/fission
cd fission
go install
pushd ../fission-bundle
./build.sh
docker build -t $FISSION_IMAGE_FQN .
gcloud docker -- push $FISSION_IMAGE_FQN
popd
pushd ../logger/fluentd
./build.sh
docker build -t $FISSION_FLUENTD_IMAGE_FQN .
gcloud docker -- push $FISSION_FLUENTD_IMAGE_FQN
cd ../../
helm upgrade --set "image=$FISSION_IMAGE,imageTag=$VERSION,pullPolicy=IfNotPresent,analytics=false,routerExternalDnsHostname=*.tfi.gcp.tesserai.com.,logger.fluentdImage=$FISSION_FLUENTD_IMAGE_FQN" harping-panda charts/fission-all
)


install_name_tool \
    -add_rpath /usr/local/cuda/lib \
    $(which python3.6)
