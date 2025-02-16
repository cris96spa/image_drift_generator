from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All the settings for the application.
    """

    # Debug mode
    debug: bool

    # Data path
    data_path: str
    competition_data: str
    menu_path: str
    codice_galattico_dir_path: str
    codice_galattico_path: str
    dish_mapping: str
    dishes_json: str
    chefs_json: str
    licenses_json: str
    techniques_json: str
    entities_path: str

    misc_path: str
    manuale_cucina_path: str
    distanze_path: str

    # Vectorstore paths
    vectorstore_relative_path: str
    debug_vectorstore_relative_path: str

    # Knowledge base path
    knowledge_base_path: str

    # Dataset path
    dataset_path: str

    # Embeddings model name
    embedding_provider: LLMProvider
    embeddings_model_name: str

    model_provider: LLMProvider

    # Model settings
    model_temperature: float = 0.0

    openai_model_name: str | None = None
    google_model_name: str | None = None

    ibm_model_name: str | None = None

    ibm_project_id: str | None = None
    ibm_endpoint_url: str | None = None

    model_config = SettingsConfigDict(
        env_file='.rag.settings',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='allow',
    )
    # MongoDB settings
    mongo_db_uri: str

    # Neo4j settings
    neo4j_url: str
    neo4j_username: str
    neo4j_password: str
