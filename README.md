# ClassSplitter

This replication package contains two datasets as well as implementation of the proposed approach.

For God class refactoring datasets, see `dataset/` directory.

For the code used in our experiment and the experiment result, see `Implementation/` directory.



## Dataset

 

The real-world God classes and their solutions are summarized with two CSV documents: 

* The `GodClassRefactorDictionary.csv` documents the refactoring examples (presented in folder `god_class_refactor_example`. This dataset has been used for the evaluation in our experiments.

* The `GodClassRefactorForPositionStudyDictionary.csv` documents the refactoring examples in folder `refactor_example_position`. This dataset has been used by the empirical study in our paper. 

The datasets record essential information about the commits where the refactorings had been applied. This information includes: `project name`, `commit hash`, `parent hash`, `extract type`, `origin class`, `new class`, `commit message`, and `url`. The data structure of the file is as follows:

![data_catalog](https://github.com/ClassSplitter/ClassSplitter/assets/146154120/ab5e4e98-e1b5-4fab-8b64-22715938c6f0)

 

Under the folders `god_class_refactor_example` and `refactor_example_position`, we present details for each of the discovered refactorings with a single CSV file. Information available in such files includes: 

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

Here is a screen shot of a CSV file:

![d_refactor_data_example](https://github.com/ClassSplitter/ClassSplitter/assets/146154120/89604c92-ba24-4354-a1ad-9b4c35929c16)

 

## Implementation

 

The implementation of the ClassSplitter method can be found in folder `ClassSplitter/`.

The manual ranking results from participants can available in `human-evaluation.xlsx`.

The raw code for the experiment is available in folder `experienments/`.

The results of the experim
