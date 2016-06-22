#!/bin/bash



sudo apt-get update

##############################
# locale
#sudo apt-get install language-pack-en
sudo locale-gen en_US.UTF-8
sudo dpkg-reconfigure locales
export LANG=en_US.UTF-8



####################################
# tools
#sudo apt-get install clang
sudo apt-get install -y htop 

###############################################
# git
git config user.name "reveriel"
git config user.email "reveriel@hotmail.com"


######################################
# shell
sudo apt-get install zsh
#sudo su
#chsh ubuntu /bin/zsh
#su ubuntu
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"


#########################################
# vim
# a vimrc simpler than spf13
git clone git://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_basic_vimrc.sh
#./spf13_install.sh 
