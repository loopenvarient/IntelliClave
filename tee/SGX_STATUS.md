##command     
docker run hello-world
sudo apt-get install -y gramine
gramine-direct --version
|
|
|
Environment: WSL2
Mode: gramine-direct
Status: Verified setup complete



##command
tee/hello_gramine/hello.py
tee/hello_gramine/hello.manifest.template
.
gramine-manifest hello.manifest.template hello.manifest
gramine-direct python3 hello.py
|
|
|
gramine-direct only



##command
grep -m1 flags /proc/cpuinfo | grep -o sgx
ls /dev/sgx* 2>/dev/null
uname -r
|
|
|
WSL2 confirmed
SGX availability (likely ❌ in WSL2)
gramine-direct mode used



##command
gramine-manifest pytorch_test.manifest.template pytorch_test.manifest
gramine-direct python3 pytorch_test.py
(and for benchmarks)
evaluate and result the benchamark baseline
.
Never re-run baseline again (important for consistency)



K8s working and YAML structured
