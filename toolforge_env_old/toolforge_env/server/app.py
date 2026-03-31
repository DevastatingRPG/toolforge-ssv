import logging
import uvicorn
from openenv.core.env_server import create_app

from toolforge_env.models import ToolForgeAction, ToolForgeObservation
from toolforge_env.server.toolforge_environment import ToolForgeEnvironment

logger = logging.getLogger(__name__)

# Create the app with web interface and README integration
app = create_app(
    ToolForgeEnvironment, 
    ToolForgeAction, 
    ToolForgeObservation, 
    env_name="toolforge_env"
)

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Main entry point for running the server.
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
