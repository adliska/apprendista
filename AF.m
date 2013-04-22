classdef AF
    properties
        desc
    end % properties

    methods
        function af = AF(description)
            af.desc = description;
        end % constructor
    end % methods
    
    enumeration
        Linear('linear')
        Sigmoid('sigmoid')
    end % enumeration
end
