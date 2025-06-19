
from pathlib import Path
import re
import json
import os
import logging
from typing import Optional, Dict, Any
from ruamel.yaml import YAML
yaml = YAML()
yaml.default_flow_style = False 
yaml.indent(mapping=2, sequence=4, offset=2)

from rag.knext.common.env import env, DEFAULT_HOST_ADDR
from rag.kag.common.llm.llm_config_checker import LLMConfigChecker
from rag.kag.common.vectorize_model.vectorize_model_config_checker import VectorizeModelConfigChecker
from rag.knext.common.utils import copyfile, copytree
from rag.knext.project.client import ProjectClient
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class projectCreator:
    def __init__(self):
        self.namespace_pattern = re.compile(r"^[A-Z][A-Za-z0-9]{0,15}$")

    def validate_namespace(self, namespace: str) -> None:
        """验证命名空间格式"""
        if not namespace:
            raise ValueError("Namespace is required")
        if not self.namespace_pattern.match(namespace):
            raise ValueError(
                f"Invalid namespace: {namespace}. Must start with uppercase letter, "
                "only contain alphanumeric characters, and have max length of 16"
            )

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            return yaml.load(Path(config_path).read_text() or "{}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    
    def create_project(
        self,
        config_path: str,
        tmpl: Optional[str] = None,
        delete_cfg: bool = False
    ) -> Path:
        
        config = self.load_config(config_path)
        project_config = config.get("project", {})
        namespace = project_config.get("namespace")
        name = project_config.get("name", namespace)
        host_addr = project_config.get("host_addr")

        self.validate_namespace(namespace)
        tmpl = tmpl or "default"

        # 校验LLM配置和向量化配置
        llm_config_checker = LLMConfigChecker()
        vectorize_model_config_checker = VectorizeModelConfigChecker()
        
        llm_config = config.get("chat_llm", {})
        vector_config = config.get("vectorizer", {})
        try:
            llm_config_checker.check(json.dumps(llm_config))
            dim = vectorize_model_config_checker.check(json.dumps(vector_config))
            config["vectorizer"]["vector_dimensions"] = dim
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise

        # 创建远程项目
        project_id = None
        if host_addr:
            project = ProjectClient(host_addr).create(
                name=name,
                namespace=namespace,
                config=json.dumps(config)
            )
            project_id = project.id if project else None

        # 渲染模板
        project_dir = self._render_template(
            namespace=namespace,
            tmpl=tmpl,
            id=project_id,
            with_server=(host_addr is not None),
            name=name,
            project_id=project_id,
            host_addr=host_addr,
            config_path=config_path,
            delete_cfg=delete_cfg
        )
        
        current_dir = os.getcwd()
        os.chdir(project_dir)
        update_project(project_dir)
        os.chdir(current_dir)

        # 清理配置
        if delete_cfg and Path(config_path).exists():
            Path(config_path).unlink()

        logger.info(
            f"Project created successfully at {project_dir}\n"
            f"Namespace: {namespace}"
        )
        return project_dir


    def _render_template(self,namespace: str, tmpl: str, **kwargs):
        config_path = kwargs.get("config_path", None)
        project_dir = Path(namespace)
        if not project_dir.exists():
            project_dir.mkdir()

        import rag.kag.templates.project

        src = Path(rag.kag.templates.project.__path__[0])
        copytree(
            src,
            project_dir.resolve(),
            namespace=namespace,
            root=namespace,
            tmpl=tmpl,
            **kwargs,
        )

        import rag.kag.templates.schema

        src = Path(rag.kag.templates.schema.__path__[0]) / f"{{{{{tmpl}}}}}.schema.tmpl"
        if not src.exists():
            import logging
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"No such schema template: {tmpl}.schema.tmpl")

        dst = project_dir.resolve() / "schema" / f"{{{{{tmpl}}}}}.schema.tmpl"
        copyfile(src, dst, namespace=namespace, **{tmpl: namespace})

        tmpls = [tmpl, "default"] if tmpl != "default" else [tmpl]
        # find all .yaml files in project dir
        config = yaml.load(Path(config_path).read_text() or "{}")
        project_id = kwargs.get("id", None)
        config["project"]["id"] = project_id
        config_file_path = project_dir.resolve() / "kag_config.yaml"
        with open(config_file_path, "w") as config_file:
            yaml.dump(config, config_file)
        return project_dir

def update_project(proj_path):
    if not proj_path:
        proj_path = env.project_path
    client = ProjectClient(host_addr=env.host_addr)

    llm_config_checker = LLMConfigChecker()
    vectorize_model_config_checker = VectorizeModelConfigChecker()
    llm_config = env.config.get("chat_llm", {})
    vectorize_model_config = env.config.get("vectorizer", {})
    try:
        llm_config_checker.check(json.dumps(llm_config))
        dim = vectorize_model_config_checker.check(json.dumps(vectorize_model_config))
        env._config["vectorizer"]["vector_dimensions"] = dim
    except Exception as e:
        import logging
        # 配置基础日志设置（只需在程序开始时配置一次）
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # 实际错误输出代码（替换原click.secho）
        logging.error(f"Error occurred: {e}", exc_info=True)


    logger.info(f"project id: {env.id}")
    client.update(id=env.id, config=json.dumps(env._config))
    import logging
    # 配置日志记录器（通常在程序初始化时执行一次）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 替换原click.secho的成功提示
    logging.info(f"Project [{env.name}] with namespace [{env.namespace}] was successfully updated from [{proj_path}].")
