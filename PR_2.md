Before working on this branch, include your previous work from step 1 (which have been merged to the main branch) in your branch:

```bash
git checkout 2/from_notebooks_to_modules
git pull origin master
git push
```

> Note: you can also use a `rebase` if you are familiar with this command.
The goal of this PR is to:

- [ ] Adapt the training code of your model that you have written in the notebook to python scripts.
- [ ] Prepare the deployment part with pickle
- [ ] Setup a CI that runs at least one linter and a code formatter

Files to be modified: the `src/modelling` folder

___

*TODO: delete this markdown file before merging the pull request*
