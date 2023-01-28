import warnings
from tqdm import tqdm
from typing import Union, List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Index

warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (19, 10)})


class Analysis:
    """
    Analysis(dataframe)

    shows required plots.

    Attributes
    ----------
    make_plot:
        makes required plot
    combine_plots:
        combines 2 or more plots in one plot
    make_pair_plot:
        make pair plot of the data frame

    Parameters
    ----------
    dataframe : pandas.DataFrame
        data to plot.
    """

    __PLOTS = {
        "bar": sns.barplot,
        "boxen": sns.boxenplot,
        "heatmap": sns.heatmap,
        "box": sns.boxplot,
        "count": sns.countplot,
        "feature_box": pd.DataFrame.boxplot,
        "dis": sns.displot,
        "dist": sns.distplot,
        "ecdf": sns.ecdfplot,
        "hist": sns.histplot,
        "joint": sns.jointplot,
        "kde": sns.kdeplot,
        "line": sns.lineplot,
        "pair": sns.pairplot,
        "custom_pair": None,
        "point": sns.pointplot,
        "reg": sns.regplot,
        "rel": sns.relplot,
        "resid": sns.residplot,
        "scatter": sns.scatterplot,
        "strip": sns.stripplot,
        "swarm": sns.swarmplot,
        "violin": sns.violinplot,
        "top_count": sns.countplot,
    }
    __SPECIAL_PLOTS = [
        "count",
        "top_count",
        "feature_box",
        "dist",
        "ecdf",
        "kde",
        "pair",
        "custom_pair",
        "heatmap",
    ]
    __NUMERIC_PLOTS = ["resid", "reg", "kde", "dist"]

    def __init__(self, dataframe: pd.DataFrame):
        """
        :param dataframe: pandas dataframe
        """
        # initializing all variables
        self.numeric_dtypes = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]
        self.grouped_df = None
        self.add_rug = None
        self.hue = None
        self.agg = None
        self.rotation = None
        self.title_font = None
        self.title = None
        self.x = None
        self.y = None
        self.plot_name = None
        self.df = dataframe
        self.top = None

    def __copy__(self):
        """
        Returns a copy of the class's current state
        it is used to combine 2 or more plots.
        """
        # creating a copy of current state
        duplicate = Analysis(self.df)
        duplicate.grouped_df = self.grouped_df
        duplicate.add_rug = self.add_rug
        duplicate.hue = self.hue
        duplicate.agg = self.agg
        duplicate.rotation = self.rotation
        duplicate.title_font = self.title_font
        duplicate.title = self.title
        duplicate.x = self.x
        duplicate.y = self.y
        duplicate.plot_name = self.plot_name
        return duplicate

    def _check_plot_name_(self, numeric=False):
        """
        checks for plot_name if it is valid plot or numeric plot
        :param numeric: to check for numeric plot
        :return:
        """
        if numeric:
            # checking if plot is numeric
            assert (
                self.plot_name not in self.__NUMERIC_PLOTS
            ), f"{self.plot_name} plot only takes numeric x"
        else:
            # checking if plot is valid
            assert (
                self.plot_name in self.__PLOTS
            ), f"{self.plot_name} is not a valid name valid plot names are {[i for i in self.__PLOTS.keys()]}"

    def _show_plots_(self, y):
        """
        adds labels and title to plot and show them
        :param y: taking y separately as some plots don't have y values
        """
        # adding x and y labels
        plt.ylabel(y)
        plt.xlabel(self.x)
        # rotating x labels
        plt.xticks(rotation=self.rotation)
        # adding title
        y_for_title = f" and {y}" if y is not None else ""

        # joint plot has small shape so setting its tile to none for now will be updated later
        if self.plot_name == "joint":
            self.title = ""
        plt.title(
            self.title
            if self.title is not None
            else f"analysis of {self.x}{y_for_title} by {self.plot_name} plot",
            fontsize=self.title_font,
        )
        # showing plot
        plt.show()

    def _add_rug_(self, y):
        """
        adds rug_plot to the plot
        :param y: y-axis value of plot
        """
        # adding rug plot based on condition
        sns.rugplot(data=self.df, x=self.x, y=y,
                    hue=self.hue) if self.add_rug else None

    @staticmethod
    def _convert_text_to_number_(text: plt.Text):
        """
        converts plt.Text element to integer
        :param text: plt.Text element
        :return: number from the text
        """
        # converting text to string
        text = str(text)
        # splitting text around the '
        text = text.split("'")
        # by splitting by ' we will get 3 elements in our list first will be before it
        # second will be the number between it
        # third will be the closing brackets
        # getting the number from list
        number = int(text[-2])
        return number

    def _draw_aggregates_(self, y):
        """
        draws the required aggregates on the plot
        :param y: y-axis value of plot
        """
        if not self.agg:
            # if there is no agg then returning from function
            return None
        # getting required columns of df
        df = self.grouped_df.agg(self.agg).reset_index()
        try:
            # seaborn uses matplotlib at backend so xtick starts from 0 thus it offsets the 2 plots by 1 unit that's why
            # some second plots position are reduced by unit of starting value for numeric labels
            # getting label of first value from the xticks
            text = plt.xticks()[1][0]
            # subtracting it to make plots even
            df[self.x] -= (
                self._convert_text_to_number_(text)
                if self.plot_name
                in ["bar", "boxen", "box", "point", "strip", "swarm", "violin"]
                else 0
            )
        except:
            pass
        # plotting a line plot for every aggregate for y
        [
            sns.lineplot(
                x=df[self.x],
                y=df[y][ag],
                legend="brief",
                linewidth=1.5,
                label=f"{y}_{ag}",
            )
            for ag in self.agg
        ]

    def _count_plot_(self):
        """
        creates a count plot in descending order
        :return: None
        """
        # getting value counts of x
        data = self.df[self.x].value_counts(ascending=False).reset_index()
        # converting labels to string so that it won't be sorted when plotted
        data["index"] = data["index"].apply(lambda x: str(x))
        # if user only wants top then cutting df to top 10 values
        if self.plot_name == "top_count":
            data = data.iloc[: self.top]
        # displaying a bar plot of x
        sns.barplot(data=data, x="index", y=self.x, order=data["index"])

    def _plot_pair_plot_(self):
        """
        draw a pair plot of data
        :param args: additional arguments
        :return: None
        """
        # draws pair plot of dataframe
        sns.pairplot(self.df)
        # plt.title("pair plot of data frame" if not self.title else self.title)
        plt.show()

    def _feature_box_(self):
        """
        shows box plot of required features from data
        :return: None
        """
        try:
            self.df[self.x if self.x is not None else self.df.columns].boxplot()
        except:
            self.df[[self.x] if self.x is not None else self.df.columns].boxplot()
        plt.show()

    def convert_categorical_data(
        self, label_dict: dict = None
    ) -> tuple[DataFrame, dict[Any, Index], set[Any]]:
        """
        it converts categorical columns to numeric columns
        :param label_dict: previous mapping to the elements of columns
        :return: updated pandas dataframe, mapping, names of categorical columns
        """
        if label_dict is None:
            label_dict = {}
        df = self.df.copy()
        # initializing categorical column as a set as it might be repeated in the below two steps
        cat_cols = set()
        # automatically converting categorical cols to numerical
        # selecting all cols except numeric types
        cat_cols = cat_cols.union(df.select_dtypes(
            exclude=self.numeric_dtypes).columns)
        for cat_col in cat_cols:
            # converting them to numeric and storing their mapping
            classes = pd.Categorical(df[cat_col])
            df[cat_col] = classes.codes
            label_dict[cat_col] = classes.categories
        return df, label_dict, cat_cols

    def make_pair_plot(
        self,
        columns_use: List[str] = None,
        convert_categorical: bool = False,
        categories_boundary: int = None,
        label_dict=None,
        rotation: int = 0,
        save_path: str = None,
        title: str = None,
        fig_size: Tuple[int, int] = None,
        return_data: bool = False,
        show_plot: bool = True,
        verbose=False,
    ) -> Union[None, Tuple[pd.DataFrame, dict]]:
        """
        This function creates a pair plot of the input dataframe.
        Pair plot is a good way to visualize the relationship between all the features of a dataframe.

        Parameters:
        :param columns_use: columns to use
        :param convert_categorical: it will automatically convert non-numeric columns to categorical-columns
        :param categories_boundary: convert numeric categorical columns to categorical
        :param label_dict: dictionary containing mapping of the categorical columns
        :param rotation: x_ticks rotation
        :param save_path: path to save the plot
        :param title: title of the SUP-PLOT
        :param fig_size: size of the figure
        :param return_data: if user wants converted_df and its dictionary
        :param show_plot: to display drawn plot
        :param verbose: to show details of process

        :return: None|tuple(pd.DataFrame,dict)
        """
        # initializing label_dictionary for categorical values
        if label_dict is None:
            label_dict = {}

        # converting cat_vals into numeric vals
        cat_cols = set()
        if convert_categorical:
            df, label_dict, cat_cols = self.convert_categorical_data(
                label_dict)
        else:
            df = self.df.copy()
        df = df[columns_use] if columns_use else df
        # selecting only numeric types
        df = df.select_dtypes(self.numeric_dtypes)

        columns = df.columns

        # getting categorical cols names from numerical columns according to given boundary
        if categories_boundary:
            cat_cols = cat_cols.union(
                df.columns[(df.nunique() <= categories_boundary)])

        if verbose:
            print("Categorical columns =", cat_cols)

        # making a fig suitable for plot_number of columns
        fig_size = (
            (8 * len(columns), 8 * len(columns)) if fig_size is None else fig_size
        )
        title_size = 30 if fig_size is None else 5
        figure, ax = plt.subplots(len(columns), len(columns), figsize=fig_size)

        # setting title of plot
        figure.suptitle(
            title if title is not None else "PAIR PLOT",
            fontsize=title_size * len(columns),
        )
        if verbose:
            pbar = tqdm(range(len(columns)**2))
        # running loop for every col as a row
        for row, rx in zip(columns, ax):
            # running loop for every col as a column
            for col, cx in zip(columns, rx):
                if col in cat_cols:
                    # setting rotation of xticks
                    cx.set_xticklabels(labels=[], rotation=rotation)
                labeled_plot = False

                if (
                    col == row
                ):  # if the col is a categorical column then it shows its value count
                    if col in cat_cols:
                        sns.countplot(
                            x=df[col].apply(
                                lambda x: label_dict[col][x] if col in label_dict else x
                            ),
                            ax=cx,
                        )
                    else:  # it shows distribution of data if column is not a categorical one
                        sns.histplot(
                            df[row], stat="density", alpha=0.5, ax=cx, color="darkblue"
                        )
                        sns.kdeplot(df[row], ax=cx, color="darkblue")
                        # sns.histplot(df[row], ax=cx,kde=True,color='darkblue')
                        cx.set(xlabel="", ylabel="")

                elif col in cat_cols:
                    # if cols are different but both are categorical then showing counts of column with hue set as row
                    if row in cat_cols:
                        sns.countplot(
                            x=df[col].apply(
                                lambda x: label_dict[col][x] if col in label_dict else x
                            ),
                            hue=df[row].apply(
                                lambda x: label_dict[row][x] if row in label_dict else x
                            ),
                            ax=cx,
                        )
                    # if only the col is a categorical column then showing a box plot between them
                    else:
                        sns.boxplot(
                            y=df[row],
                            x=df[col].apply(
                                lambda x: label_dict[col][x] if col in label_dict else x
                            ),
                            ax=cx,
                        )
                # if only the row is a categorical column then showing a violin plot between them
                elif row in cat_cols:
                    sns.violinplot(
                        y=df[row].apply(
                            lambda x: label_dict[row][x] if row in label_dict else x
                        ),
                        x=df[col],
                        orient="horizontal",
                        ax=cx,
                    )
                    labeled_plot = True
                else:  # if both row and col are numerical than showing a regression plot between them
                    sns.regplot(df, x=col, y=row, ax=cx, color="darkblue")

                # Some plots are already labeled but if it's not labeled, we make sure to label it
                # for better readability and understanding of the data.
                if not labeled_plot:
                    cx.set(xlabel=col, ylabel=row)
                # showing processes info if required
                if verbose:
                    pbar.update()
        if save_path:
            print(f"saving plot as {save_path}")
            plt.savefig(save_path, dpi=300)
        if show_plot:
            print("showing plot")
            plt.show()
        else:
            plt.close()
        # returning data and label dictionary
        if return_data:
            return df, label_dict

    def _heatmap_(self):
        """
        makes heatmap of dataframe based on the correlation of columns of df
        it also shows corr with categorical values
        :return: None
        """
        # making correlations
        converted_data, labled_dict, cat_cols = self.convert_categorical_data()
        corr = converted_data.corr()
        # making heatmap
        self.__PLOTS["heatmap"](corr, annot=True, cmap="Greys")
        # showing plot
        self._show_plots_(None)

    def _create_special_plots_(self):
        """
        shows the plots with only x data
        """
        if self.plot_name == "pair":
            return self._plot_pair_plot_()
        elif self.plot_name == "custom_pair":
            return self.make_pair_plot(convert_categorical=True, verbose=True,title=self.title, rotation=self.rotation)
        elif self.plot_name == "heatmap":
            return self._heatmap_()
        elif self.plot_name == "feature_box":
            return self._feature_box_()
        elif self.plot_name in ["count", "top_count"]:
            return self._count_plot_()
        try:
            # plotting data
            self.__PLOTS[self.plot_name](
                x=self.df[self.x], color="cyan" if self.plot_name in [
                    "count"] else None
            )
        except ValueError:
            # above line will raise a value error if data given to it is not numeric for some plots
            # checking if plot is numeric
            self._check_plot_name_(numeric=True)
        # adding rug based on condition
        self._add_rug_(None)

    def _create_plot_(self, y, name=None):
        """
        creates the plots object with given data
        :param y: y-axis value
        :param name: name of plot
        :return:
        """
        # if plot is a facet plot then
        if self.plot_name in ["dis", "rel"]:
            self.__PLOTS[self.plot_name](
                x=self.x,
                y=y,
                hue=self.hue,
                data=self.df,
                height=10,  # facet plots have height and aspect ratio
                aspect=1.5,
                label=name,
            ).fig.suptitle(
                # setting its title
                f"Analysis of {self.x} based on {y} by {self.plot_name}_plot",
                fontsize=self.title_font,
            )
        else:
            try:
                # creating plot for Y if it is not a facet plot
                self.__PLOTS[self.plot_name](
                    x=self.x, y=y, hue=self.hue, label=name, data=self.df,
                )
            except TypeError:
                try:
                    # some plots dont except hue and some label so type error is raised
                    self.__PLOTS[self.plot_name](
                        x=self.x, y=y, label=name, data=self.df,
                    ) if self.plot_name not in [
                        "bar",
                        "box",
                        "boxen",
                        "point",
                        "joint",
                    ] else self.__PLOTS[
                        self.plot_name
                    ](
                        x=self.x, y=y, hue=self.hue, data=self.df,
                    )

                except TypeError:
                    # for reg and resid plot type error is raised if data is not numeric
                    # so checking and raising appropriate error
                    self._check_plot_name_(numeric=True)
            except ValueError:
                # for some plots Value error is raised if data is not numeric do raising appropriate error
                self._check_plot_name_(numeric=True)
            # adding a rug according to condition
            self._add_rug_(y)

            # adding agg if there is any
            self._draw_aggregates_(y)

    def make_plot(
        self,
        plot_name: str,
        x: Union[List, str] = None,
        y: Union[List, str] = None,
        title: str = None,
        title_font: int = 20,
        rotation: int = 45,
        agg: list = None,
        hue: str = None,
        top: int = 10,
        add_rug: bool = False,
        create_copy=False,
    ):
        """
        shows the plot with given data
        :param plot_name: name of plot
        :param x: x label its list form is only valid in feature_plot
        :param y: y label
        :param title: title of plot
        :param title_font: font size f title
        :param rotation: rotation of x labels
        :param agg: aggregates
        :param hue: hue label
        :param top: value for top labels in count plot
        :param add_rug: rug condition
        :param create_copy: to return copy of class
        :return: None
        """
        # updating values of all variables
        self.plot_name = plot_name
        self.x = x
        self.y = y
        self.title = title
        self.title_font = title_font
        self.rotation = rotation
        self.agg = agg
        self.top = top
        self.hue = hue
        self.add_rug = add_rug

        # Validating plot_name
        self._check_plot_name_()

        # checking if the plot is a special one
        if self.plot_name in self.__SPECIAL_PLOTS:
            # creating and showing special plot
            self._create_special_plots_()
            if self.plot_name not in [
                "heatmap",
                "pair",
                "custom_pair",
                "feature_box",
            ]:  # these plots have there separate labels
                self._show_plots_(None)

        else:  # if plot is a normal one
            # determining value of y if it is a str or list and grouping data by x and selecting y
            if self.y:
                self.y = [self.y] if isinstance(self.y, str) else self.y
                # grouping df by x
                self.grouped_df = self.df.groupby(self.x)[self.y]
            else:
                self.grouped_df = None
                self.y = [None]

            # going over y labels
            for _y in self.y:
                # creating plot for y's
                self._create_plot_(_y)
                # displaying plots
                self._show_plots_(_y) if self.plot_name not in [
                    "dis",
                    "rel",
                ] else plt.show()
        # If necessary, returning a copy of all variables
        if create_copy:
            return self.__copy__()

    @staticmethod
    def combine_plot(plots: list, x=None, y=None, title=None):
        """
        combines the plot objects it works best if given plots have same x labels

        :param plots: list of analysis objects to combine
        :param x: title of x axis
        :param y: title of y axis
        :param title: title of graph
        """
        # creating all plots
        for plot in plots:
            plot._create_plot_(plot.y[0], name=plot.y[0])
        # adding title and showing the plot
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=35)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
