---
layout: page
title: To-do-list
permalink: /todolist/
parent: Python
---

# To do list via CLI

File contains a script that implements a basic task list **with text interface**. 

Database contains: 
* hash id
* name of the task
* deadline
* description

It is necessary to changethe the path in the file ex2_database.py to run it.

The next stage is to change the path in ex2.py and run via console with appropriate args:
```
python ex2.py --type [add|update|list|remove]
```

In **add** parse, **--name** argument is obligatory (str), but --deadline (in format DDMMYYYY) and --description (str) are optional. Script return hash id (MD5: calculating by concatenate name+deadline+description), if adding to database has been made successfully.
```
python ex2.py --type add --name Send a mail --description Post office
```

Returned value: eba1a42c2412faacc98f386716d25998

In **update** parse **--task_hash** is necessary to be typed (in the database, the id is the primary key). --name, --deadline, --description are optional argument. Type what you want to change in the database, eg.
```
python ex2.py --type update --description Car cleaning --deadline 16032020 --task_hash 5ff64bccdd09d2c6a66149191d20d6a0
```

In such a case, it doesn't affect the name value in the database. It will change the MD5 hash and return the updated value.

In **--remove**, it is necessary to type **--task_hash**, eg.
```
python ex2.py --type remove --task_hash 5ff64bccdd09d2c6a66149191d20d6a0
```
In **--list**, it is obligatory to set **--[all|today]**. 
The first returned column is hash id, the second name of the task, third deadline, and fourth description. 
```
python ex2.py --type list --today
```

[Link button](http://example.com/){: .btn }<i class="fa fa-car"></i>

[Link button](http://example.com/){: .btn .btn-purple }
[Link button](http://example.com/){: .btn .btn-blue }
[Link button](http://example.com/){: .btn .btn-green }

[Link button](http://example.com/){: .btn .btn-outline }

In case of trouble, there is also a --help option.
