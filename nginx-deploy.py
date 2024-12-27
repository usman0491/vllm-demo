from ray import serve
from ray.serve.handle import DeploymentHandle
import subprocess

# These imports are used only for type hints:
from typing import Dict
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment
class NginxDeployment:
    def __init__(self):
        self.container_id = subprocess.check_output(
            [
                "docker", "run", "-d", "--rm",
                "-p", "80:80",
                "nginx:latest"
            ]
        ).decode("utf-8").strip()
        print(f"Nginx container started with ID: {self.container_id}")

    def __del__(self):
        subprocess.run(["docker", "stop", self.container_id], check=True)
        print(f"Nginx container stopped: {self.container_id}")

    async def __call__(self, request):
	    return JSONResponse({"status": "Nginx container is running."})

@serve.deployment
class RootRouter:
    def __init__(self):
        self.nginx = NginxDeployment.bind()

    async def __call__(self, request):
	    return JSONResponse({"message": "Welcome to the Ray Serve Nginx App!"})



deployment_graph = RootRouter.bind()
