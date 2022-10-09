import os
import configparser
from pathlib import Path
from typing import Dict, Union, Optional

from ru_encoders_benchmark.logging_handler import get_logger


# use as:
# class SomeClass:
# def __init__(self, config_path: str)
#    self._config_path = config_path
#    init_config(config_path)
#    ...

class Configurator:
    """
    Manages the configuration of data paths within the benchmark.
    """

    def __init__(self, config_path: str, create_if_not_exists: bool):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.create_if_not_exists = create_if_not_exists
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    def init_config(self, **kwargs):
        """
        Initializes the config object.
        """
        if not os.path.exists(self.config_path):
            if self.create_if_not_exists:
                self.config = self.create_config(self.config_path, **kwargs)
            else:
                self._logger.error(f"Config file by path provided: '{self.config_path}' does not exists!"
                                   "Check file path or select option to create it.")
                raise FileNotFoundError(f"Config file by path provided: '{self.config_path}' does not exists!"
                                        "Check file path or select option to create it.")
        else:
            self.config.read(self.config_path)
        self._logger.info("Config file successfully initialized.")

    @classmethod
    def create_config(cls, config_path: str, save_to_file: bool = True,
                      configuration_parameters: Optional[Dict[str, str]] = None) -> configparser.ConfigParser:
        """
        Create a config file
        """
        config = configparser.ConfigParser()

        current_dir = Path().resolve().absolute()
        default_configuration_parameters = {
            "STS": str(current_dir / "STS"),
            "XNLI": str(current_dir / "XNLI"),
            "PI": str(current_dir / "ParaPhraserPlus_PI"),
            "SA": str(current_dir / "SentiRuEval2016"),
            "TI": str(current_dir / "OKMLCup_toxic"),
            "II": str(current_dir / "Inappropriateness"),
            "IC": str(current_dir / "IntentsClassification"),
            "PG": str(current_dir / "Paraphrase-NMT-Leipzig"),
            "NER1": str(current_dir / "FactRuEval2016")
        }

        config.add_section("basic_paths")
        config.set("basic_paths", "root_dir", str(current_dir))
        config.set("basic_paths", "output_dir", str(current_dir / "output"))
        config.set("basic_paths", "data_dir", str(current_dir / "data"))

        if configuration_parameters is not None:
            default_configuration_parameters.update(configuration_parameters)

        # Tasks relates datasets
        config.add_section("data_paths")
        for param_name, param_value in default_configuration_parameters.items():
            config.set("data_paths", param_name, param_value)

        if save_to_file:
            with open(str(config_path), "w") as config_file:
                config.write(config_file)

        return config

    def check_setting_existence(self, section: str, setting: str) -> bool:
        """
        Indicates whether the named section & setting is present in the configuration.
        If the given section exists, and contains the given setting, return True;
        otherwise return False.

        :param section: a requested section of config;
        :type section: string;

        :param setting: a requested setting from section of config;
        :type setting: string;

        :return: a bool flag.
        """
        if (not self.config.has_section(section)) \
                or (not self.config.has_option(section, setting)):
            self._logger.error(f"Setting '{setting}' in section '{section}'"
                               "do not exists in project configuration!")
            return False

    def get_setting(self, section: str, setting: str) -> Optional[str]:
        """
        Print out a setting.
        :param section:  a requested section of config;
        :type section: string;

        :param setting: a desired setting from section of config;
        :type setting: string;

        :return: settings value or None, if provided setting name
        do not exists in current project configuration.
        """
        if self.check_setting_existence(section, setting):
            return None
        return self.config.get(section, setting)

    def update_setting(self, section: str, setting: str,
                       value: Union[str, Path],
                       resave: bool = False):
        """
        Update a setting in file.
        """
        if self.check_setting_existence(section, setting):
            return None
        try:
            self.config.set(section, setting, str(value))
        except TypeError as e:
            self._logger.error(f"Provided new setting value cannot be cased to string!")
        if resave:
            with open(str(self.config_path), "w") as config_file:
                self.config.write(config_file)

    def delete_setting(self, section: str,
                       setting: str,
                       resave: bool = False):
        """
        Delete a setting in file
        """
        if self.check_setting_existence(section, setting):
            return None
        self.config.remove_option(section, setting)
        if resave:
            with open(str(self.config_path), "w") as config_file:
                self.config.write(config_file)


if __name__ == "__main__":
    path = "set_locations.ini"
    configurator = Configurator(path, create_if_not_exists=True)
    configurator.init_config()
