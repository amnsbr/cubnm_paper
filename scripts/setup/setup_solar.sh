#!/bin/bash

cd $(dirname $0)

PROJECT_DIR=$(realpath "$(pwd)/../..")
TOOLS_DIR="${PROJECT_DIR}/tools"

cd $TOOLS_DIR

wget https://www.nitrc.org/frs/download.php/12469/solar-eclipse-9.0.0-dynamic-ubuntu.zip
unzip solar-eclipse-9.0.0-dynamic-ubuntu.zip -d solar_src

cd solar_src
./install_solar.sh $TOOLS_DIR/solar $TOOLS_DIR/solar

# add solar to PATH in .bashrc if not already present
SOLAR_PATH=$(realpath "$TOOLS_DIR/solar")
BASHRC="$HOME/.bashrc"

# Create the export line
EXPORT_LINE="export PATH=\"\$PATH:${SOLAR_PATH}\""

# Check if the path is already in .bashrc
if ! grep -qF "$SOLAR_PATH" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Added by solar installation script" >> "$BASHRC"
    echo "$EXPORT_LINE" >> "$BASHRC"
    echo "Solar path added to .bashrc"
    echo "Please run 'source ~/.bashrc' or restart your shell to update PATH"
else
    echo "Solar path already exists in .bashrc"
fi