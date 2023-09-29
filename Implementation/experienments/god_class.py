import pandas as pd
import openai
import pandas.core.frame
import random
import networkx as nx


openai.api_key = ''


class DictHandler:

    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.data_frame: pandas.core.frame.DataFrame = pd.read_csv(dict_path, delimiter=',', header=0, encoding="MacRoman")
        self.index = self.data_frame['index'].to_list()

    def get_class(self, index=None):
        if index is None:
            index = random.randint(0, len(self.data_frame))
            try:
                new_class = GodClass(pd.DataFrame(self.data_frame.iloc[index]).T)
                return new_class
            except FileNotFoundError:
                print("err: %s not found." % str(self.data_frame.at[index, 'data path']))
                return self.get_class()
        else:
            try:
                new_class = GodClass(pd.DataFrame(self.data_frame.iloc[self.index.index(index)]).T)
                return new_class
            except FileNotFoundError:
                print("err: %s not found." % str(self.data_frame.at[self.index.index(index), 'data path']))
                return None

    def get_classes(self):
        classes = []
        for i in self.index:
            new_class = self.get_class(i)
            if new_class is not None:
                classes.append(new_class)
        return classes

    def get_filtered_classes(self):
        filtered_classes = []
        for class_ in self.get_classes():
            if class_.get_checked() is not None and class_.get_checked():
                filtered_classes.append(class_)
        return filtered_classes

    def get_filtered_class(self, index):
        new_class = self.get_filtered_classes()[index]
        return new_class


class GodClass:

    def __init__(self, data: pandas.core.frame.DataFrame):
        self.data = data
        self.data_path = str(data.at[self.data.iloc[0].name, 'data path'])
        self.data_frame: pandas.core.frame.DataFrame = \
            pd.read_csv(self.data_path, delimiter=',', header=0, encoding="MacRoman")

    def __len__(self):
        return len(self.data_frame)

    def get_gpt_respond(self):
        print("\n============================ getting %s gpt respond. ============================" % str(self.data_path))
        respond = ["" for i in range(len(self.data_frame))]
        for i in range(len(self.data_frame)):
            print("in %s getting %dth respond out of %d, respond:" % (self.data_path, i, len(self.data_frame)))
            if self.data_frame.at[i, 'type'] == "Method":
                try:
                    # content = f"read code that contains a method:\n```java\n{self.data_frame.at[i, 'full text']}\n```\ngive me a function summary description for the code in less than 3 sentences."
                    content = f"Please give me a function summary in less than 3 sentences for the following source code:\n```java\n{self.data_frame.at[i, 'full text']}\n```"
                    messages = [
                        {"role": "user",
                         "content": content},
                    ]
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=messages,
                        temperature=0,
                    )
                    respond[i] = response['choices'][0]['message']['content']
                except Exception as e:
                    print(e)
            else:
                respond[i] = self.data_frame.at[i, 'commit']
            print(respond[i])
        self.data_frame['code summary gpt4'] = respond

    def to_csv(self, filename=None):
        if filename is None:
            filename = self.data_path
        self.data_frame.to_csv(filename, index=False)

    def get_method_number(self):
        num = 0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'type'] == "Method":
                num += 1
        return num

    def get_field_number(self):
        num = 0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'type'] == "Field":
                num += 1
        return num

    def get_removed_percentage(self):
        total_method_number = self.get_method_number()
        extract_method_number = 0.0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'removed'] and self.data_frame.at[i, 'type'] == "Method":
                extract_method_number += 1
        if total_method_number > 0:
            return extract_method_number / total_method_number
        else:
            return 0.0

    def get_removed_field_percentage(self):
        total_field_number = self.get_field_number()
        extract_field_number = 0.0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'removed'] and self.data_frame.at[i, 'type'] == "Field":
                extract_field_number += 1
        if total_field_number > 0:
            return extract_field_number / total_field_number
        else:
            return 0.0

    def get_name(self):
        return str(self.data.at[self.data.iloc[0].name, 'origin class'])

    def get_type(self):
        return self.data.at[self.data.iloc[0].name, 'extract type']

    def get_index(self):
        return int(self.data.at[self.data.iloc[0].name, 'index'])

    def get_last_index(self):
        return int(self.data.at[self.data.iloc[0].name, 'last index'])

    def get_url(self):
        return str(self.data.at[self.data.iloc[0].name, 'url'])

    def get_checked(self):
        try:
            text = self.data.at[self.data.iloc[0].name, 'checked']
            if text == 'Good':
                return True
            elif text == 'Bad':
                return False
            else:
                return None
        except KeyError:
            return None

    # def draw_class(self):
    #     graph = nx.Graph()
    #     nodes = [i for i in range(1, len(self.data_frame + 1))]
    #     edges = []
    #     weights = []
    #     for i in range(len(self.methods)):
    #         for j in range(i + 1, len(self.methods)):
    #             edges.append((self.methods[i], self.methods[j]))
    #             weights.append(self.get_method_coupling(i, j)[0])
    #     for i in range(len(nodes)):
    #         graph.add_node(nodes[i])
    #     for i in range(len(edges)):
    #         graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])


