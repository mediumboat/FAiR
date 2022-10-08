# FAiR
Code repository for "Mitigating Popularity Bias for Users and Items with Fairness-centric Adaptive Recommendation" that will be published on Transaction on Information Systems (TOIS).
## 1. Description for each file
	dataloader.py : some ultilities to load the raw data and make it in proper format for our model.
	Model.py : the implementation of FAiR model and training steps.
	Evaluator.py : some ultilities to rank items and calculate various metrics for evaluation of effective and fairness.
	main.py : the example codes for training and testing the model.

## 2. Requirements (Environment)
	python >= 3.6
	tensorflow >= 2.2.0
  	numpy
  	pandas
  	tqdm 
  	pathlib 


## 3. How to run

- (1) Prepare the dataset (see below).
- (2) Read the description of hyper-parameters and configure them in main.py (`line 72-110`)
- (3) Run "python main.py".



## 4. Datasets

The datasets could be downloaded from the links in the paper (a sample data is provided in the repository), readers should place them in proper directory and modify `data_path` variable in dataloader.py.

It is also possible to use reader's own dataloader to produce proper formatted data for training and testing. Generally, our model requires the following input variables (see main.py and Model.py for more details):

- `n_users`/`n_items`: integer, the total numbers of users and items respectively.
- `x_train`/`x_val`/`x_test`: a numpy matrix with has 2 columns, the first column is the user_id and the second is the item_id. Note that all user and item ids should start from 0.
- `y_train`/`y_val`/`y_test`: a numpy matrix with 1 column which stores the rating of corresponding `x` variable. Note that `y` should have same number of rows as corresponding `x`.
- `user_group`/`item_group`: a numpy matrix with 1 column which stores the group of each user/item (the element is 0 or 1). Note that the number of rows should be `n_users` or `n_items`.
- `avg_rating`: the reference rating scale r0 as described in the paper. It is a numpy matrix with 1 column which stores the averaged rating of each item from all training users. Note that the number of rows should be `n_items`.


## 5. Other instructions

The implementation of different components of FAiR can be found in Model.py (i.e., class `Filter`, `Discriminator` and `BaseRecModel`). The feature initialization (i.e., `pertrain()` in class `FAiR`) and adversarial training process (i.e., `adv_train()` in class `FAiR`) are also in the same file. Readers can have their own implementations on those components using other kinds of backbones.

## 6. Contact

For any questions, please contact me (zzliu[DOT]2020[AT]phdcs[DOT]smu[DOT]edu[DOT]sg)
