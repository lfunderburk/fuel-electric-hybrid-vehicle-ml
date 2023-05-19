### Docker setup

Publishing image

```
docker build -t <application>:latest -f <Dockerfile-name> .
docker tag <application>:latest lgfunderburk/<application>:latest
docker push lgfunderburk/<application>:latest
```

Where

`<application>` is one of `myapi`, `mydashapp`, `mypipeline` and
`<Dockerfile-name>` is one of `Dockerfile.api`, `Dockerfile.dash` and `Dockerfile.pipe`

# Running image

```
 docker build -t vehicleapi:latest -f Dockerfile .
docker run -it --rm -p 8000:8000 vehicleapi
 ```