# Path to your oh-my-zsh installation.
export ZSH=/Users/guoxing/.oh-my-zsh

# Set name of the theme to load.
# Look in ~/.oh-my-zsh/themes/
# Optionally, if you set this to "random", it'll load a random theme each
# time that oh-my-zsh is loaded.
ZSH_THEME="agnoster"

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion. Case
# sensitive completion must be off. _ and - will be interchangeable.
 HYPHEN_INSENSITIVE="true"

# Uncomment the following line to disable bi-weekly auto-update checks.
 #DISABLE_AUTO_UPDATE="true"

# Uncomment the following line to change how often to auto-update (in days).
 #export UPDATE_ZSH_DAYS=13

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
 #ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
 COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# The optional three formats: "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
# HIST_STAMPS="mm/dd/yyyy"

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load? (plugins can be found in ~/.oh-my-zsh/plugins/*)
# Custom plugins may be added to ~/.oh-my-zsh/custom/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(git
zsh-syntax-highlighting)


# User configuration

export PATH="/Library/Frameworks/Python.framework/Versions/3.4/bin:/usr/local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/texbin"
#export MANPATH=/usr/local/man:$MANPATH

export PATH=$HOME/.bin:$PATH
export PATH=$PATH:~/llvm36/bin
export PATH=/usr/local/bin:$PATH
export PATH=$PATH:/usr/local/texlive/2015/bin/x86_64-darwin
export PATH=$HOME/.local/bin:$PATH      # for stack for haskell
export PATH=$HOME/.cabal/bin:$PATH    # for cabal of haskell

# setting for SICP
export PATH=$PATH:/Applications/Racket\ v6.3/bin

# stratego/xt
export PATH=/opt/strategoxt/v0.17/bin:$PATH
export PATH=/opt/aterm/v2.5/bin:$PATH
export PATH=/opt/sdf2-bundle/v2.4/bin:$PATH

##
export DYLD_LIBRARY_PATH=$HOME/llvm36/lib

# oracle database
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HOME/db/lib/instantclient_11_2-3
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HOME/db/lib/instantclient_11_2

export PATH=$PATH:$HOME/db/bin
alias sqlplus="rlwrap sqlplus"



## java
export JAVA_HOME="$(/usr/libexec/java_home)"
export LuceneDir=$HOME/project/lucene-5.3.1

export CLASSPATH=$CLASSPATH:$LuceneDir/demo/lucene-demo-5.3.1.jar
export CLASSPATH=$CLASSPATH:$LuceneDir/analysis/common/lucene-analyzers-common-5.3.1.jar
export CLASSPATH=$CLASSPATH:$LuceneDir/core/lucene-core-5.3.1.jar
export CLASSPATH=$CLASSPATH:$LuceneDir/queryparser/lucene-queryparser-5.3.1.jar

export CLASSPATH=$CLASSPATH:~/project/algo/sedgewick/algs4.jar

## mysql
export CLASSPATH=$HOME/project/mysql/mysql-connector-java-5.1.39-bin.jar:$CLASSPATH


## beatify man 
export MANPAGER="col -b | mvim -v -c 'set ft=man ts=8 nomod nolist nonu' \
    -c 'nnoremap i <nop>' \
    -c 'nnoremap a <nop>' \
    -c 'nnoremap c <nop>' \
    -c 'nnoremap s <nop>' \
    -c 'nnoremap <Space> <C-f>' \
    -c 'noremap q :quit<CR>' -"
source $ZSH/oh-my-zsh.sh

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='mvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# ssh
# export SSH_KEY_PATH="~/.ssh/dsa_id"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"

alias s="ls -GF"
alias l="ls -CF"
alias la="ls -aCFG"
alias ll="ls -als"
alias rm="rm -i"

alias vim="mvim -v"

#alias rm="echo please use del, so you won't regret"
alias del="rmtrash"

alias cnpm="npm --registry=https://registry,npm.taobao.org \
    --cache=$HOME/.npm/.cache/cnpm/ \
    --disturl=https://npm.taobao.org/dist \
    --userconfig=$HOME/.cnpmrc"



alias subl='open -a "Sublime Text"'
alias em="emacsclient -t"
##

alias scheme="racket -i -p neil/sicp -l xrepl"
# -i enables interactive mode. repl mode.
# -p neil/sicp enables use of the SICP package.
# -l xrepl, enables Racket’s extended REPL mode.


#ssh for complier thery
eval `ssh-agent -s`
ssh-add ~/.ssh/gx_ustc

# openMP


#autojump
[[ -s $(brew --prefix)/etc/profile.d/autojump.sh ]] && . $(brew --prefix)/etc/profile.d/autojump.sh
