from pydantic_settings import BaseSettings


class PlatformSettings(BaseSettings):
    platform_url: str

    @classmethod
    def how_to_settings(cls) -> str:
        settings_fileds = cls.model_fields
        platform_url = settings_fileds['platform_url'].default
        required_keys = [name for name, field in settings_fileds.items() if field.is_required()]
        optional_keys = [name for name, field in settings_fileds.items() if not field.is_required()]
        # 生成设置文档
        prefix = cls.model_config.get('env_prefix', '')
        platform_name = cls.__name__.replace('Settings', '')
        return f"""# Platform
{platform_name}

# Required Environment Variables
{[(prefix + name).upper() for name in required_keys]}

# Optional Environment Variables
{[(prefix + name).upper() for name in optional_keys]}

You can get more information from this link: {platform_url}

tips: You can also set these variables in the .env file, and generate will automatically load them."""
