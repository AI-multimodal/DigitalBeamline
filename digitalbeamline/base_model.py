from abc import ABC, abstractmethod, abstractproperty

from digitalbeamline.utils import create_home_directory


class ABCModel(ABC):

    @abstractproperty
    def name(self):
        """The name of the model. Should be a file name with something like a
        .pt extension.

        Returns
        -------
        str
        """

        ...

    @abstractproperty
    def permalink(self):
        """The url to the permanent link where the model is stored. For
        example, a Zenodo DOI. This will only be used if the model name is not
        found in the local $HOME/.DigitalBeamline directory.

        Returns
        -------
        str
        """

        ...

    @abstractmethod
    def fetch(self):
        """Pulls the model from its online location to the local storage
        location."""

        ...

    @abstractmethod
    def load(self):
        """Loads the model from its local location into memory."""

        ...

    @abstractmethod
    def info(self):
        """Prints the model card information in a well-formatted way."""

        ...

    def featurizer(self, x):
        return x

    @abstractmethod
    def model(self, x):
        ...

    def postprocessor(self, x):
        return x

    def __call__(self, x):
        x = self.featurizer(x)

