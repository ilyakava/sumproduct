# commands from: https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
# an alternative would be to run: make -e PLATFORM=test release
PLATFORM=pypi

package:
	python setup.py sdist
	python setup.py bdist_wheel

release:
	python setup.py register -r $(PLATFORM)
	python setup.py sdist upload -r ${PLATFORM}
	python setup.py bdist_wheel upload -r ${PLATFORM}
ifeq (test, $(PLATFORM))
	$(info now do: pip install -i https://testpypi.python.org/pypi sumproduct)
endif

clean:
	rm -rf build/ dist/ sumproduct.egg-info/

readme:
	pandoc -s readme.md -o README.rst
