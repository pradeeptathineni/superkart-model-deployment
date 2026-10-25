FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Remove the unused Yarn source whose missing key blocks Docker's apt setup
# SuperKart has no Yarn dependency; other package signature checks stay enabled
RUN rm -f /etc/apt/sources.list.d/yarn.list

# devcontainer.json adds SSH support for Codespaces CLI status checks
