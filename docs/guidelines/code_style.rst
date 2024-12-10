Git hooks
=========

`Git hooks <https://git-scm.com/book/ms/v2/Customizing-Git-Git-Hooks>`_ are a convenient tool
to execute code on specific git events, e.g. before a commit.

Feel free to use the following git hooks (this is Jakobs current setup for the cyten repo)

1. pre-push: put the following contents ins `.git/hooks/pre-push`

    ```bash
    #!/bin/bash

    STOP_PUSH=false

    # 1) Check for forbidden patterns in the patch / diff of any of the commits to be pushed
    # ======================================================================================

    # "@{push}" is the target of "git push" and "@" is HEAD 
    # Filter only added (A), copied (C), modified (M) files
    for file in $(git diff --name-only --diff-filter=ACM @{push} @); do
        # note: only the last grep should be quiet (-q)!
        #       the first one should return the matches, the last only the exit code
        if git show :0:"$file" | grep -E "^\+" | grep -Eq "^[<>=]{7}"; then
            echo -e "\033[1;31mLeftover conflict markers in $file\033[0m"
            STOP_PUSH=true
        fi
    done

    # 2) Check for forbidden patterns in the commit messages
    # ======================================================================================

    for hash in $(git rev-list @{push}..@); do
        MESSAGE=$(git log -n 1 --pretty=format:%s "$hash")
        SHORTHASH=$(git log -n 1 --pretty=format:%h "$hash")
    done

    # 3) Lint
    # ======================================================================================

    # Run custom linting script
    if python tests/linting/python_linting.py &> /dev/null
    then
        echo "Custom linting passed"
    else
        python tests/linting/python_linting.py
        STOP_PUSH=true
    fi

    # Run flake8 linter
    if python -m flake8 . &> /dev/null
    then
        echo "flake8 passed"
        else
        python -m flake8 .
        STOP_PUSH=true
    fi

    # 4) If there was any error then stop push
    # ======================================================================================

    if $STOP_PUSH; then
        echo "\033[1;31mPush blocked.\033[0m Use '--no-verify' to circumvent the hook and push anyway."
        exit 1
    fi
    ```
    
2. pre-commit: put the following contents ins `.git/hooks/pre-commit`

    ```bash
    #!/bin/bash

    STOP_COMMIT=false

    # Check for forbidden patterns in the patch / diff
    # ======================================================================================

    # Loop over staged files, filter diff to only add added, copied, modified files
    for file in $(git diff --staged --name-only --diff-filter=ACM); do
    # check for leftover conflict markers in *whole file*
    # note: only the last grep should be quiet (-q)!
    #       the first one should return the matches, the last only the exit code
    if git show :0:"$file" | grep -E "^\+" | grep -Eq "^[<>=]{7}"; then
        echo -e "\033[1;31mLeftover conflict markers in $file\033[0m"
        STOP_COMMIT=true
    fi
    if git diff --staged $file | grep -E "^\+" | grep -Eq "breakpoint()|set_trace()"; then
        echo -e "\033[1;31mDebugging breakpoint in $file\033[0m"
        STOP_COMMIT=true
    fi
    done

    # Lint
    # ======================================================================================

    # Run custom linting script
    if python tests/linting/python_linting.py &> /dev/null
    then
    echo "Custom linting passed"
    else
    python tests/linting/python_linting.py
    STOP_COMMIT=true
    fi

    # Run flake8 linter
    if python -m flake8 . &> /dev/null
    then
    echo "flake8 passed"
    else
    python -m flake8 .
    STOP_COMMIT=true
    fi

    # If there was any error then stop commit
    # ======================================================================================

    if $STOP_COMMIT; then
    echo -e "\033[1;31mCommit blocked.\033[0m Use '--no-verify' to circumvent the hook and commit anyway."
    exit 1
    fi
    ```

3. On MacOS only, note that the builtin ``grep`` command does not support the features used above.
   Use the GNU version from `homebrew <https://formulae.brew.sh/formula/grep>` instead.
   Note that ``brew`` installs the command under the different name ``ggrep`` by default,
   so either adjust the hook accordingly or add ``$HOMEBREW_PREFIX/opt/grep/libexec/gnubin``
   to your ``$PATH``.
