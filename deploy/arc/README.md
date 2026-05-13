# Self-hosted GPU runners on k3s via ARC

This directory contains everything needed to provision a GitHub Actions
self-hosted runner pool on the local k3s cluster using the
[Actions Runner Controller (ARC)](https://github.com/actions/actions-runner-controller)
`gha-runner-scale-set` chart — the modern, officially supported runner
deployment model.

The runner pool name is `mini-sglang-gpu`, which is the `runs-on` value used by
`.github/workflows/gpu-tests.yml`.

## Prerequisites on the k3s node

1. **NVIDIA driver** installed on the host. Verify with `nvidia-smi`.
2. **nvidia-container-toolkit** + containerd configured. For k3s, edit
   `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl` to add the
   `nvidia` runtime, then `systemctl restart k3s`.
3. **NVIDIA device plugin** as a DaemonSet:
   ```bash
   kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml
   kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
   # Should print the number of GPUs on the node.
   ```
4. **Helm** v3 installed locally (used to deploy ARC).

## One-time install

### 1. Install the ARC controller

```bash
helm install arc \
  --namespace arc-systems --create-namespace \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

### 2. Create the auth secret

Generate a fine-grained PAT with `actions:write`, `administration:write`,
`metadata:read` on this repo. Then:

```bash
kubectl create namespace arc-runners
kubectl create secret generic arc-github-secret \
  --namespace arc-runners \
  --from-literal=github_token=ghp_REPLACE_ME
```

(For long-lived setups, prefer a GitHub App — see ARC docs.)

### 3. Build & push the runner image

The default ARC runner image has no CUDA. Build the GPU-capable variant:

```bash
docker build -f deploy/arc/runner.Dockerfile \
  -t ghcr.io/chakpongchung/mini-sglang-runner:latest .
docker push ghcr.io/chakpongchung/mini-sglang-runner:latest
```

If your GHCR is private, also create an image-pull secret in `arc-runners`
and reference it via `template.spec.imagePullSecrets` in `values.yaml`.

### 4. Deploy the runner scale set

```bash
helm install mini-sglang-gpu \
  --namespace arc-runners \
  -f deploy/arc/values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

The release name (`mini-sglang-gpu`) is what the workflow's `runs-on`
references. Do not rename it without also editing
`.github/workflows/gpu-tests.yml`.

## Verifying the setup

```bash
# 1. Controller running
kubectl get pods -n arc-systems

# 2. Listener for our scale set running, zero idle runners (minRunners=0)
kubectl get pods -n arc-runners
# Expect:  mini-sglang-gpu-...-listener   1/1 Running

# 3. GitHub sees the runner scale set
# → Repo Settings → Actions → Runners → "Self-hosted runners" tab
#   should show "mini-sglang-gpu" with status "Idle" or "Active".
```

## Triggering a CI run end-to-end

1. Push a PR against `main` (or amend any commit on the PR branch). The
   workflow's `pull_request` trigger fires on the GitHub side.
2. ARC's listener observes the queued job and spawns a runner pod:
   ```bash
   kubectl get pods -n arc-runners -w
   # mini-sglang-gpu-runner-xxxxx  0/1 Pending → ContainerCreating → Running
   ```
3. The job advances through the workflow's steps. Tail logs from the
   pod or the Actions UI:
   ```bash
   kubectl logs -n arc-runners -l app.kubernetes.io/component=runner -f --tail=200
   ```
4. The pod is destroyed when the job ends (`minRunners: 0`).

## Common failure modes

| Symptom | Fix |
|---|---|
| Runner pod stuck `Pending` with `0/2 nodes available: insufficient nvidia.com/gpu` | Device plugin not installed, or GPU already claimed by another pod. Run `kubectl describe node` and check `Allocatable.nvidia.com/gpu`. |
| Pod runs but `nvidia-smi` step fails | Either `runtimeClassName: nvidia` is needed (uncomment in `values.yaml`) or `NVIDIA_VISIBLE_DEVICES` isn't propagating — verify the device plugin DaemonSet pod logs on the node. |
| Install step fails with `flashinfer`/`sgl_kernel` JIT errors | CUDA toolkit not reachable inside the pod. The provided Dockerfile uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` which includes nvcc; if you build off a `runtime` (not `devel`) base, JIT compiles will fail. |
| Listener pod CrashLoopBackOff, logs mention `401 Unauthorized` | Secret `arc-github-secret` has a bad/expired token, or token lacks the required scopes. Recreate it. |
| `kernel/test_tensor.py` fails with `CUDA error: invalid device ordinal` | Pod only has 1 GPU. The test hard-codes `cuda:1`. Bump `nvidia.com/gpu` from 1 to 2 in `values.yaml`. |
