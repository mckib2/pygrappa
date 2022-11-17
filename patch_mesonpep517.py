import pathlib
import mesonpep517


if __name__ == "__main__":
    buildapi = pathlib.Path(mesonpep517.__file__).parent / "buildapi.py"
    with open(buildapi, "r") as fp:
        contents = fp.read()
    contents = contents.replace(".decode('utf-8').strip('\\n')", ".decode('utf-8').strip('\\n\\r')")
    contents = contents.replace("abi = get_abi(python)", "abi = get_abi(sys.executable)")
    with open(buildapi, "w") as fp:
        fp.write(contents)
    print("Patched buildapi.py for Windows!")

    pyproject = pathlib.Path(__file__).parent / "pyproject.toml"
    with open(pyproject, "r") as fp:
        contents = fp.read()
    contents += "\n\n[tools.pip]\ndisable-isolated-build = true\n"
    with open(pyproject, "w") as fp:
        fp.write(contents)
