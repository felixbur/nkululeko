class DefineBase:
    @classmethod
    def _assert_has_attribute_value(cls, value):
        valid_values = cls._attribute_values()
        if value not in valid_values:
            raise BadValueError(value, valid_values)

    @classmethod
    def _attribute_values(cls):
        attributes = inspect.getmembers(cls, lambda x: not inspect.isroutine(x))
        return sorted(
            [
                a[1]
                for a in attributes
                if not (a[0].startswith("__") and a[0].endswith("__"))
            ]
        )


class Header(DefineBase):
    HEADER_RESULTS = "Results"
    HEADER_EXPLORE = "Data exploration"
