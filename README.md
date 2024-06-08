# ClassSplitter

This replication package contains dataset and code related to our research paper.

For God class refactoring dataset, see `dataset/` directory.

For the code used in our experiment and the experiment result, see `Implementation/` directory.

## Dataset

The real-world God classes and their refactoring are organized by two CSV-formated catalog: 

* The `GodClassRefactorDictionary.csv` catalog documents the refactoring examples in `god_class_refactor_example` folder, which were used for evaluation in our experiments.
* The `GodClassRefactorForPositionStudyDictionary.csv` catalog documents the refactoring examples in `refactor_example_position` folder, which were used to validate the correlation between code position and god class refactoring. 

The catalog files record essential information about the commits where the refactorings take place. This information includes: `project name`, `commit hash`, `parent hash`, `extract type`, `origin class`, `new class`, `commit message`, and `url`. The data structure of the catalogs is as follows:
![data_catalog](https://github.com/ClassSplitter/ClassSplitter/assets/146154120/861f7013-c8ba-47aa-928a-7cf3727fef00)

The refactorings are presented as individual CSV files. Each file represents a class before refactoring, with each row in the file recording an entity from the original class. These rows contain the features of the entities, as well as their refactoring information. The file paths are documented in the `data path` column in the catalog. The recorded features of the entities include:

| tag                  | description                                                  |
| -------------------- | ------------------------------------------------------------ |
| name                 | name of the entity, including parameter types for methods    |
| type                 | type of the entity, Field or Method or MemberClass           |
| inner invocations    | for field entities, methods in the same class who visited the field, split with space |
| external invocations | for field entities, methods in other classes  who visited the field, split with space |
| calls                | for method entities, the methods called by the method, split with space |
| visits               | for method entities, the fields visited by the method, split with space |
| length               | length of the entity, by char                                |
| lines                | number of lines the entity has                               |
| modifier             | java modifier number of the entity                           |
| annotation           | annotation before the entity (if exist)                      |
| full text            | full content of the entity                                   |
| removed              | whether the entity is extracted from the origin class in refactoring, mark TRUE for extracted entities, FALSE for kept entities |

and the data structure of the refactoring examples is as follows:
![d_refactor_data_example](https://github.com/ClassSplitter/ClassSplitter/assets/146154120/89604c92-ba24-4354-a1ad-9b4c35929c16)


## Implementation

the code implementation of the ClassSplitter method can be found in `ClassSplitter/` folder.

the ranking data of participants for refactoring result can be checked in `human-evaluation.xlsx`.

the raw code for the experiment is in `experienments/` folder.

the main experiment results can be checked in `experienments/experienment_result.ipynb`.
