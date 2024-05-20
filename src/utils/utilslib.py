from typing import Any, List
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

def generate_box_plot(data: List[Any], xlabel: str, ylabel: str, box_color: str) -> None:
    """This function will create a box plot for a given data.
        
        Params:
            data: List of values
            xlabel: Label for x-axis
            ylabel: Label for y-axis

        Returns:
            None.
        """
    fig = plt.figure(figsize =(5, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    boxprops = dict(facecolor = box_color, edgecolor = 'black')
    medianprops = dict(color='black')
    flierprops = dict(marker = 'o', markerfacecolor = "#F55252", markersize = 8, linestyle = 'none')
    box = plt.boxplot(data, patch_artist = True, boxprops = boxprops, medianprops = medianprops, flierprops = flierprops)
    plt.show()


def comparative_box_plot(data: List[Any], xlabel: str, ylabel: str, box_color: str) -> None:
    """This function will create a box plot for a given data.
        
        Params:
            data: List of values
            xlabel: Label for x-axis
            ylabel: Label for y-axis

        Returns:
            None.
        """
    fig = plt.figure(figsize =(5, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    boxprops = dict(facecolor = box_color, edgecolor = 'black')
    medianprops = dict(color='black')
    flierprops = dict(marker = 'o', markerfacecolor = "#F55252", markersize = 8, linestyle = 'none')
    box = plt.boxplot(data, patch_artist = True, boxprops = boxprops, medianprops = medianprops, flierprops = flierprops)
    plt.show()

def preprocess_text(text: str) -> List[str]:
    """ This function is used to preprocess text by converting characters to lowercase
        and removes all characters which are not alphanumeric.
    
    Params:
        text: A string value to preprocess.

    Returns:
        List[str]: List of tokens after preprocessing.
    """
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum()]